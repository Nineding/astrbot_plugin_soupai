import asyncio
import json
import os
import threading
from datetime import datetime
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.message_components import At
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.utils.session_waiter import (
    SessionController,
    SessionFilter,
    session_waiter,
)

from .judging import LEVEL_COMMENTS, JudgeEngine, Judgment


# 线程安全的题库管理基类
class ThreadSafeStoryStorage:
    """线程安全的题库管理基类，支持持久化使用记录"""

    def __init__(self, storage_name: str, data_path=None):
        self.storage_name = storage_name
        self.data_path = data_path
        self.used_indexes: set[int] = set()
        self.lock = threading.Lock()  # 线程锁
        self.usage_file = (
            self.data_path / f"{storage_name}_usage.json" if self.data_path else None
        )
        self.load_usage_record()

    def load_usage_record(self):
        """从文件加载使用记录"""
        if not self.usage_file:
            self.used_indexes = set()
            return

        try:
            if self.usage_file.exists():
                with open(self.usage_file, encoding="utf-8") as f:
                    self.used_indexes = set(json.load(f))
                logger.info(
                    f"从 {self.usage_file} 加载了 {len(self.used_indexes)} 个使用记录"
                )
            else:
                self.used_indexes = set()
                logger.info(f"使用记录文件不存在，创建新的记录: {self.usage_file}")
        except Exception as e:
            logger.error(f"加载使用记录失败: {e}")
            self.used_indexes = set()

    def save_usage_record(self):
        """保存使用记录到文件"""
        if not self.usage_file:
            return

        try:
            self.usage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage_file, "w", encoding="utf-8") as f:
                json.dump(list(self.used_indexes), f, ensure_ascii=False, indent=2)
            logger.info(
                f"保存了 {len(self.used_indexes)} 个使用记录到 {self.usage_file}"
            )
        except Exception as e:
            logger.error(f"保存使用记录失败: {e}")

    def reset_usage(self):
        """重置使用记录"""
        with self.lock:
            self.used_indexes.clear()
            self.save_usage_record()
            logger.info(f"{self.storage_name} 使用记录已重置")

    def get_usage_info(self) -> dict:
        """获取使用记录信息"""
        with self.lock:
            return {
                "used": len(self.used_indexes),
                "used_indexes": list(self.used_indexes),
            }


# 游戏状态管理
class GameState:
    def __init__(self):
        self.active_games: dict[str, dict] = {}  # 群聊ID -> 游戏状态

    def start_game(self, group_id: str, puzzle: str, answer: str, **extra) -> bool:
        """开始游戏，返回是否成功"""
        if group_id in self.active_games:
            return False
        game_data = {
            "puzzle": puzzle,
            "answer": answer,
            "is_active": True,
            "qa_history": [],
            "hint_history": [],
        }
        game_data.update(extra)
        self.active_games[group_id] = game_data
        return True

    def end_game(self, group_id: str) -> bool:
        """结束游戏"""
        if group_id in self.active_games:
            del self.active_games[group_id]
            return True
        return False

    def get_game(self, group_id: str) -> dict | None:
        """获取游戏状态"""
        return self.active_games.get(group_id)

    def is_game_active(self, group_id: str) -> bool:
        """检查是否有活跃游戏"""
        return group_id in self.active_games


# 网络海龟汤管理
class NetworkSoupaiStorage(ThreadSafeStoryStorage):
    def __init__(self, network_file: str, data_path=None):
        # 初始化基类
        super().__init__("network_soupai", data_path)
        self.network_file = network_file
        self.stories: list[dict] = []
        self.load_stories()

    def load_stories(self):
        """从文件加载网络海龟汤故事"""
        try:
            if os.path.exists(self.network_file):
                with open(self.network_file, encoding="utf-8") as f:
                    self.stories = json.load(f)
                logger.info(
                    f"从 {self.network_file} 加载了 {len(self.stories)} 个网络海龟汤故事"
                )
            else:
                self.stories = []
                logger.warning(f"网络海龟汤文件不存在: {self.network_file}")
        except Exception as e:
            logger.error(f"加载网络海龟汤失败: {e}")
            self.stories = []

    def get_story(self) -> tuple[str, str] | None:
        """从网络题库获取一个故事，避免重复（线程安全）"""
        if not self.stories:
            return None

        with self.lock:
            # 获取所有可用的索引（排除已使用的）
            available_indexes = [
                i for i in range(len(self.stories)) if i not in self.used_indexes
            ]

            # 如果没有可用题目，清空已用记录，重新开始一轮
            if not available_indexes:
                logger.info("网络题库已全部使用完毕，清空记录重新开始")
                self.used_indexes.clear()
                available_indexes = list(range(len(self.stories)))
                # 立即保存重置后的状态
                self.save_usage_record()

            # 从可用索引中随机选择一个
            import random

            selected = random.choice(available_indexes)
            self.used_indexes.add(selected)

            # 保存使用记录
            self.save_usage_record()

            story = self.stories[selected]
            logger.info(
                f"从网络题库获取故事，索引: {selected}, 已使用: {len(self.used_indexes)}/{len(self.stories)}"
            )
            return story["puzzle"], story["answer"]

    def get_storage_info(self) -> dict:
        """获取网络题库信息"""
        usage_info = self.get_usage_info()
        return {
            "total": len(self.stories),
            "available": len(self.stories) - usage_info["used"],
            "used": usage_info["used"],
        }


# 存储库管理
class LocalSoupaiStorage(ThreadSafeStoryStorage):
    def __init__(self, storage_file: str, max_size: int = 50, data_path=None):
        # 初始化基类
        super().__init__("storage_soupai", data_path)
        self.storage_file = storage_file
        self.max_size = max_size
        self.stories: list[dict] = []
        self.load_stories()

    def load_stories(self):
        """从文件加载故事"""
        try:
            storage_path = (
                self.storage_file
                if isinstance(self.storage_file, str)
                else str(self.storage_file)
            )
            if os.path.exists(storage_path):
                with open(storage_path, encoding="utf-8") as f:
                    self.stories = json.load(f)
                logger.info(f"从 {storage_path} 加载了 {len(self.stories)} 个故事")
            else:
                self.stories = []
                logger.info("存储库文件不存在，创建新的存储库")
        except Exception as e:
            logger.error(f"加载故事失败: {e}")
            self.stories = []

    def save_stories(self):
        """保存故事到文件"""
        try:
            storage_path = (
                self.storage_file
                if isinstance(self.storage_file, str)
                else str(self.storage_file)
            )
            # 确保目录存在
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, "w", encoding="utf-8") as f:
                json.dump(self.stories, f, ensure_ascii=False, indent=2)
            logger.info(f"保存了 {len(self.stories)} 个故事到 {storage_path}")
        except Exception as e:
            logger.error(f"保存故事失败: {e}")

    def add_story(self, puzzle: str, answer: str) -> bool:
        """添加故事到存储库"""
        with self.lock:
            if len(self.stories) >= self.max_size:
                # 移除最旧的故事
                self.stories.pop(0)
                logger.info("存储库已满，移除最旧的故事")

            story = {
                "puzzle": puzzle,
                "answer": answer,
                "created_at": datetime.now().isoformat(),
            }
            self.stories.append(story)
            self.save_stories()
            logger.info(f"添加新故事到存储库，当前存储库大小: {len(self.stories)}")
            return True

    def get_story(self) -> tuple[str, str] | None:
        """从存储库获取一个故事，避免重复（线程安全）"""
        if not self.stories:
            return None

        with self.lock:
            # 获取所有可用的索引（排除已使用的）
            available_indexes = [
                i for i in range(len(self.stories)) if i not in self.used_indexes
            ]

            # 如果没有可用题目，清空已用记录，重新开始一轮
            if not available_indexes:
                logger.info("本地存储库已全部使用完毕，清空记录重新开始")
                self.used_indexes.clear()
                available_indexes = list(range(len(self.stories)))
                # 立即保存重置后的状态
                self.save_usage_record()

            # 从可用索引中随机选择一个
            import random

            selected = random.choice(available_indexes)
            self.used_indexes.add(selected)

            # 保存使用记录
            self.save_usage_record()

            story = self.stories[selected]
            logger.info(
                f"从本地存储库获取故事，索引: {selected}, 已使用: {len(self.used_indexes)}/{len(self.stories)}"
            )
            return story["puzzle"], story["answer"]

    def get_storage_info(self) -> dict:
        """获取存储库信息"""
        usage_info = self.get_usage_info()
        return {
            "total": len(self.stories),
            "max_size": self.max_size,
            "available": self.max_size - len(self.stories),
            "used": usage_info["used"],
            "remaining": len(self.stories) - usage_info["used"],
        }


# 自定义海龟汤存储
class CustomSoupaiStorage(ThreadSafeStoryStorage):
    def __init__(self, storage_file: str, data_path=None):
        # 初始化基类
        super().__init__("custom_soupai", data_path)
        self.storage_file = storage_file
        self.stories: list[dict] = []
        self.load_stories()

    def load_stories(self):
        """从文件加载自定义故事"""
        try:
            storage_path = (
                self.storage_file
                if isinstance(self.storage_file, str)
                else str(self.storage_file)
            )
            if os.path.exists(storage_path):
                with open(storage_path, encoding="utf-8") as f:
                    self.stories = json.load(f)
                logger.info(
                    f"从 {storage_path} 加载了 {len(self.stories)} 个自定义海龟汤故事"
                )
            else:
                self.stories = []
                logger.info("自定义海龟汤文件不存在，创建新的存储库")
        except Exception as e:
            logger.error(f"加载自定义海龟汤失败: {e}")
            self.stories = []

    def save_stories(self):
        """保存自定义故事到文件"""
        try:
            storage_path = (
                self.storage_file
                if isinstance(self.storage_file, str)
                else str(self.storage_file)
            )
            # 确保目录存在
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, "w", encoding="utf-8") as f:
                json.dump(self.stories, f, ensure_ascii=False, indent=2)
            logger.info(
                f"保存了 {len(self.stories)} 个自定义海龟汤故事到 {storage_path}"
            )
        except Exception as e:
            logger.error(f"保存自定义海龟汤失败: {e}")

    def add_story(self, puzzle: str, answer: str) -> bool:
        """添加自定义故事到存储库"""
        with self.lock:
            story = {
                "puzzle": puzzle,
                "answer": answer,
                "created_at": datetime.now().isoformat(),
            }
            self.stories.append(story)
            self.save_stories()
            logger.info(f"添加新自定义海龟汤故事，当前存储库大小: {len(self.stories)}")
            return True

    def get_story(self) -> tuple[str, str] | None:
        """从自定义存储库获取一个故事，避免重复（线程安全）"""
        if not self.stories:
            return None

        with self.lock:
            # 获取所有可用的索引（排除已使用的）
            available_indexes = [
                i for i in range(len(self.stories)) if i not in self.used_indexes
            ]

            # 如果没有可用题目，清空已用记录，重新开始一轮
            if not available_indexes:
                logger.info("自定义存储库已全部使用完毕，清空记录重新开始")
                self.used_indexes.clear()
                available_indexes = list(range(len(self.stories)))
                # 立即保存重置后的状态
                self.save_usage_record()

            # 从可用索引中随机选择一个
            import random

            selected = random.choice(available_indexes)
            self.used_indexes.add(selected)

            # 保存使用记录
            self.save_usage_record()

            story = self.stories[selected]
            logger.info(
                f"从自定义存储库获取故事，索引: {selected}, 已使用: {len(self.used_indexes)}/{len(self.stories)}"
            )
            return story["puzzle"], story["answer"]

    def get_storage_info(self) -> dict:
        """获取自定义存储库信息"""
        usage_info = self.get_usage_info()
        return {
            "total": len(self.stories),
            "used": usage_info["used"],
            "remaining": len(self.stories) - usage_info["used"],
        }


# 验证结果类
class VerificationResult:
    """验证结果类"""

    def __init__(
        self,
        level: str,
        comment: str,
        is_correct: bool = False,
        *,
        judgment: Judgment | None = None,
    ):
        self.level = level
        self.comment = comment
        self.is_correct = is_correct
        self.judgment = judgment

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "comment": self.comment,
            "is_correct": self.is_correct,
        }


# 自定义会话过滤器 - 以群为单位进行会话控制
class GroupSessionFilter(SessionFilter):
    """会话过滤器，确保每个群的会话独立"""

    def __init__(self, group_id: str):
        # 为每个会话保存其所属群 ID
        self.group_id = group_id

    def filter(self, event: AstrMessageEvent) -> str:
        current_group_id = (
            event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        )
        # 仅当事件来自该群时才返回有效的会话 ID，否则返回空串避免误触发
        return self.group_id if current_group_id == self.group_id else ""


@register(
    "astrbot_plugin_soupai",
    "KONpiGG",
    "AI 海龟汤推理游戏插件，支持自动生成谜题、智能判断、验证系统、智能提示、存储库管理等功能。网络题库包含近300道海龟汤，还在持续更新中。",
    "1.1.0",
    "https://github.com/Nineding/astrbot_plugin_soupai",
)
class SoupaiPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.game_state = GameState()

        # 获取配置值
        self.generate_llm_provider_id = self.config.get("generate_llm_provider", "")
        self.judge_llm_provider_id = self.config.get("judge_llm_provider", "")
        self.review_llm_provider_id = self.config.get("review_llm_provider", "")
        self.game_timeout = self.config.get("game_timeout", 300)
        self.storage_max_size = self.config.get("storage_max_size", 50)
        self.auto_generate_start = self.config.get("auto_generate_start", 3)
        self.auto_generate_end = self.config.get("auto_generate_end", 6)
        self.puzzle_source_strategy = self.config.get(
            "puzzle_source_strategy", "network_first"
        )
        # TODO: 别名兼容处理，建议若干版本后删除
        if self.puzzle_source_strategy == "ai_first":
            self.puzzle_source_strategy = "local_first"

        # 汤底真相进度开关
        self.reveal_progress = self.config.get("reveal_progress", False)

        # 难度设置
        self.difficulty_settings = {
            "简单": {
                "limit": None,
                "accept_levels": ["完全还原", "核心推理正确"],
                "hint_limit": 10,
            },
            "普通": {
                "limit": None,
                "accept_levels": ["完全还原"],
                "hint_limit": 5,
            },
            "困难": {
                "limit": 15,
                "accept_levels": ["完全还原"],
                "hint_limit": 1,
            },
            "666开挂了": {
                "limit": 5,
                "accept_levels": ["完全还原"],
                "hint_limit": 0,
            },
        }
        self.group_difficulty: dict[str, str] = {}

        # 数据存储路径: 使用框架提供的工具获取插件数据目录
        self.data_path = StarTools.get_data_dir()
        self.data_path.mkdir(parents=True, exist_ok=True)

        # 存储库初始化延迟到 init 方法中
        self.local_story_storage = None
        self.online_story_storage = None
        self.custom_story_storage = None

        # 防止重复调用的状态
        self.generating_games = set()  # 正在生成谜题的群聊ID集合

        # 自动生成状态
        self.auto_generating = False
        self.auto_generate_task = None
        self.auto_schedule_task = None
        self._session_tasks: set[asyncio.Task] = set()

    def _ensure_story_storages(self) -> None:
        """确保题库存储被初始化。

        在某些环境下, 插件的 ``init`` 方法可能未被调用或异常退出,
        导致存储对象仍为 ``None``。为避免后续调用出现
        ``'NoneType' object has no attribute 'get_story'`` 的错误, 这里
        提供一次性惰性初始化。
        """

        if self.local_story_storage is None:
            storage_file = self.data_path / "storage_soupai.json"
            self.local_story_storage = LocalSoupaiStorage(
                storage_file, self.storage_max_size, self.data_path
            )

        if self.online_story_storage is None:
            plugin_dir = Path(__file__).resolve().parent
            network_file = plugin_dir / "network_soupai.json"
            self.online_story_storage = NetworkSoupaiStorage(
                str(network_file), self.data_path
            )

        if self.custom_story_storage is None:
            custom_file = self.data_path / "custom_soupai.json"
            self.custom_story_storage = CustomSoupaiStorage(custom_file, self.data_path)

    async def init(self, context: Context):
        """插件初始化，此时 self.data_path 可用"""
        await super().init(context)

        # 初始化存储对象
        self._ensure_story_storages()

        # 启动自动生成任务
        self.auto_schedule_task = asyncio.create_task(self._start_auto_generate())

        online_info = self.online_story_storage.get_storage_info()
        logger.info(
            f"海龟汤插件已加载，配置: 生成LLM提供商={self.generate_llm_provider_id}, 判断LLM提供商={self.judge_llm_provider_id}, 超时时间={self.game_timeout}秒, 网络题库={online_info['total']}个谜题, 本地存储库大小={self.storage_max_size}, 谜题来源策略={self.puzzle_source_strategy}, 汤底真相进度={self.reveal_progress}"
        )

    async def terminate(self):
        """Cancel owned sessions and background jobs before unloading the plugin."""
        self.game_state.active_games.clear()
        self.generating_games.clear()
        self.auto_generating = False
        tasks = set(self._session_tasks)
        tasks.update(
            task
            for task in (self.auto_generate_task, self.auto_schedule_task)
            if task is not None
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._session_tasks.clear()
        logger.info("Soupai sessions and background jobs stopped")

    def mask_answer_with_progress(self, answer: str) -> str:
        """将答案中除标点符号外的字符替换为█，用于显示真相进度"""
        import string

        # 定义标点符号（包括中文标点）
        punctuation = string.punctuation + "，。！？；：''【】（）《》、·…～"
        result = []
        for char in answer:
            if char in punctuation or char.isspace():
                result.append(char)
            else:
                result.append("█")
        return "".join(result)

    def _merge_reveal_progress(
        self,
        knowed_answer: str,
        revealed_content: str,
        true_answer: str,
    ) -> str:
        """根据 LLM 返回的待揭露片段，在当前进度中揭开对应位置。"""
        if not revealed_content or revealed_content == "未命中":
            return knowed_answer

        if len(knowed_answer) != len(true_answer):
            return knowed_answer

        merged = list(knowed_answer)
        start = 0
        has_match = False

        while True:
            idx = true_answer.find(revealed_content, start)
            if idx == -1:
                break

            has_match = True
            for pos in range(idx, min(idx + len(revealed_content), len(true_answer))):
                if merged[pos] == "█":
                    merged[pos] = true_answer[pos]
            start = idx + 1

        if has_match:
            return "".join(merged)
        return knowed_answer

    def _reveal_by_question_match(
        self,
        knowed_answer: str,
        question: str,
        true_answer: str,
        max_new_chars: int = 6,
    ) -> str:
        """当 LLM 进度更新不可用时，按提问关键词在真相中做保守揭露。"""
        import re

        if len(knowed_answer) != len(true_answer):
            return knowed_answer

        # 提取提问中的中英文/数字关键词，长度>=2 以降低误触发。
        tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", question)
        tokens = [t for t in tokens if len(t) >= 2]
        if not tokens:
            return knowed_answer

        # 优先长词，尽量命中更具体片段。
        tokens.sort(key=len, reverse=True)

        merged = list(knowed_answer)
        revealed = 0
        for token in tokens:
            start = 0
            while revealed < max_new_chars:
                idx = true_answer.find(token, start)
                if idx == -1:
                    break

                for pos in range(idx, min(idx + len(token), len(true_answer))):
                    if revealed >= max_new_chars:
                        break
                    if merged[pos] == "█":
                        merged[pos] = true_answer[pos]
                        revealed += 1
                start = idx + 1

            if revealed >= max_new_chars:
                break

        return "".join(merged)

    async def _start_auto_generate(self):
        """启动自动生成任务"""
        while True:
            try:
                now = datetime.now()
                current_hour = now.hour

                # 检查是否在自动生成时间范围内
                if self.auto_generate_start <= current_hour < self.auto_generate_end:
                    if not self.auto_generating:
                        # 检查存储库是否已满，如果已满则不启动自动生成
                        self._ensure_story_storages()
                        storage_info = self.local_story_storage.get_storage_info()
                        if storage_info["available"] <= 0:
                            logger.info(
                                f"本地存储库已满，跳过自动生成，时间: {current_hour}:00"
                            )
                            # 等待1小时后再次检查
                            await asyncio.sleep(3600)  # 1小时
                            continue

                        logger.info(f"开始自动生成故事，时间: {current_hour}:00")
                        self.auto_generating = True
                        asyncio.create_task(self._auto_generate_loop())
                else:
                    if self.auto_generating:
                        logger.info(f"停止自动生成故事，时间: {current_hour}:00")
                        self.auto_generating = False

                # 等待1小时后再次检查
                await asyncio.sleep(3600)  # 1小时
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动生成任务错误: {e}")
                await asyncio.sleep(3600)  # 出错后等待1小时再试

    async def _auto_generate_loop(self):
        """自动生成循环"""
        # 确保在运行循环前题库已初始化
        self._ensure_story_storages()
        while self.auto_generating:
            try:
                # 检查本地存储库是否已满
                storage_info = self.local_story_storage.get_storage_info()
                if storage_info["available"] <= 0:
                    logger.info("本地存储库已满，停止自动生成")
                    self.auto_generating = False
                    break

                # 生成一个故事
                puzzle, answer = await self.generate_story_with_llm()
                if puzzle and answer and not puzzle.startswith("（"):
                    self.local_story_storage.add_story(puzzle, answer)
                    logger.info("自动生成故事成功")
                else:
                    logger.warning("自动生成故事失败")

                # 等待5分钟再生成下一个
                await asyncio.sleep(300)  # 5分钟
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动生成故事错误: {e}")
                await asyncio.sleep(300)  # 出错后等待5分钟再试

    # ✅ 生成谜题和答案
    async def generate_story_with_llm(self) -> tuple[str, str]:
        """使用 LLM 生成海龟汤谜题"""

        # 根据配置获取指定的生成 LLM 提供商
        if self.generate_llm_provider_id:
            provider = self.context.get_provider_by_id(self.generate_llm_provider_id)
            if provider is None:
                logger.error(
                    f"未找到指定的生成 LLM 提供商: {self.generate_llm_provider_id}"
                )
                return "（无法生成题面，指定的生成 LLM 提供商不存在）", "（无）"
        else:
            provider = self.context.get_using_provider()
            if provider is None:
                logger.error("未配置 LLM 服务商")
                return "（无法生成题面，请先配置大语言模型）", "（无）"

        prompt = self._build_puzzle_prompt()

        try:
            logger.info("开始调用 LLM 生成谜题...")
            llm_resp: LLMResponse = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                func_tool=None,
                image_urls=[],
                system_prompt="你是一个专业的反转推理谜题创作者，专门为海龟汤游戏设计谜题。你需要创作简洁、具象、有逻辑反转的谜题，让玩家能够通过是/否提问逐步还原真相。每次创作都必须全新、原创，不能重复已有故事。",
            )

            text = llm_resp.completion_text.strip()
            logger.info(f"LLM 返回内容: {text}")

            # 尝试多种格式解析
            puzzle = None
            answer = None

            # 格式1: "题面：xxx 答案：xxx"
            if "题面：" in text and "答案：" in text:
                puzzle = text.split("题面：")[1].split("答案：")[0].strip()
                answer = text.split("答案：")[1].strip()

            # 格式2: "**题面**：xxx **答案**：xxx" (Markdown格式)
            elif "**题面**" in text and "**答案**" in text:
                puzzle = text.split("**题面**")[1].split("**答案**")[0].strip()
                if puzzle.startswith("：") or puzzle.startswith(":"):
                    puzzle = puzzle[1:].strip()
                answer = text.split("**答案**")[1].strip()
                if answer.startswith("：") or answer.startswith(":"):
                    answer = answer[1:].strip()

            # 格式3: "题面：xxx\n答案：xxx"
            elif "题面：" in text and "\n答案：" in text:
                puzzle = text.split("题面：")[1].split("\n答案：")[0].strip()
                answer = text.split("\n答案：")[1].strip()

            # 格式4: 尝试从文本中提取题面和答案
            else:
                lines = text.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # 寻找题面
                    if not puzzle and ("题面" in line or "**题面**" in line):
                        puzzle = line
                        if "：" in line:
                            puzzle = line.split("：", 1)[1].strip()
                        elif ":" in line:
                            puzzle = line.split(":", 1)[1].strip()
                        # 移除可能的Markdown标记
                        puzzle = puzzle.replace("**", "").replace("*", "").strip()

                    # 寻找答案
                    elif not answer and ("答案" in line or "**答案**" in line):
                        answer = line
                        if "：" in line:
                            answer = line.split("：", 1)[1].strip()
                        elif ":" in line:
                            answer = line.split(":", 1)[1].strip()
                        # 移除可能的Markdown标记
                        answer = answer.replace("**", "").replace("*", "").strip()

                    # 如果找到了题面但还没找到答案，继续寻找
                    elif puzzle and not answer and len(line) > 20:
                        # 可能是答案的开始
                        answer = line

            if puzzle and answer:
                # 清理答案中的多余内容
                if "----" in answer:
                    answer = answer.split("----")[0].strip()
                if "---" in answer:
                    answer = answer.split("---")[0].strip()

                logger.info(f"成功解析谜题: 题面='{puzzle}', 答案='{answer}'")
                return puzzle, answer

            logger.error(f"LLM 返回内容格式错误: {text}")
            return "生成失败", "无法解析 LLM 返回的内容"
        except Exception as e:
            logger.error(f"生成谜题失败: {e}")
            return "生成失败", f"LLM 调用出错: {e}"

    def _build_puzzle_prompt(self) -> str:
        """构建谜题生成的提示词"""
        import random

        # 丰富的主题列表，增加多样性
        themes = [
            # 🔍 人类行为与误导
            "误解他人行为的代价",
            "看似反常实则合理的选择",
            "主动伪装带来的反转",
            "隐瞒真相与道德困境",
            "他人为主角设下的圈套",
            "故意失败的计划",
            "真实动机被遮蔽",
            "道德与规则的冲突",
            # 🧠 心理博弈与控制
            "陷害与自保之间的抉择",
            "信息不对称引发的误判",
            "操控他人感知的行为",
            "主观偏见导致的误解",
            "冷静外表下的激烈动机",
            "以退为进的心理策略",
            # 🧪 现实逻辑与错觉
            "空间结构引发的错觉",
            "物品使用的误导性",
            "因果顺序的错配",
            "隐藏在日常中的意外用途",
            "非典型证据的误导",
            "时间线的巧妙安排",
            # 📍 社会环境与冲突
            "职场中的暗中博弈",
            "公众场合下的隐秘行为",
            "权力结构下的自我保护",
            "日常制度漏洞的利用",
            "面对规则边缘的选择",
            "技术被滥用的后果",
            "资源争夺下的灰色行为",
            # 🧩 特定身份与角色
            "保安不是最了解监控的人",
            "程序员的删除并非错误",
            "清洁工的观察比谁都细致",
            "老师的行为引发质疑",
            "医生做出的不寻常选择",
            "司机的路线似乎有问题",
            "演员的自毁是否另有用意",
            # 🕯 情感错位与人性
            "好意引发的巨大误会",
            "爱被误解为恶意",
            "习惯性行为暴露了真相",
            "为了他人不得不说谎",
            "逃避责任的精心设计",
            "牺牲某人换取整体安全",
        ]

        selected_theme = random.choice(themes)

        prompt = (
            f"你是一个逻辑推理谜题设计师，正在创作一个用于【海龟汤游戏】的原创谜题。\n\n"
            "【目标】：生成一个结构清晰、信息复杂、具备反差感的逻辑谜题，玩家可以通过是/否提问逐步还原真相。答案中解释的所有行为和结果，必须都在题面中有所体现或留有暗示，禁止引入题面未提及的核心行为或结果。谜题在满足以上要求的前提下，应尽可能风格多样、身份多样、行为设定独特、反转机制不重复，避免模板化创作。\n\n"
            "【题面】要求：\n"
            "1~2句话，控制在30字以内，但不能过短或单一；\n"
            "必须包含具体人物 + 至少两个具体细节或行为（如行为+环境、行为+结果、两个动作等）；\n"
            "行为必须具象明确，严禁使用抽象词、形容词、心理或情绪描述；\n"
            "必须包含异常或矛盾要素，能引发为什么？的思考；\n"
            "允许黑暗元素，如陷害、伤害、诱导、自残、掩盖证据等冷峻现实情节；\n"
            "不得使用幻想、梦境、魔法、精神病等设定；\n"
            "使用陈述句，不得使用疑问句或解释语气。\n\n"
            "【答案】要求：\n"
            "不超过200字；\n"
            "真实可实现，具有完整因果逻辑；\n"
            "至少包含两个推理层次或误导点（例如动机误导+情境误导）；\n"
            "不得出现反转在于、真相是、实际上之类的总结或解释语；\n"
            "不要使用说明性句子或教学语气；\n"
            "整体氛围可偏冷峻，但必须具备可还原性，逻辑自洽。\n"
            "答案仅用于解释题面中已有行为与结果，禁止引入题面未包含的额外关键事件或角色。\n\n"
            "参考例子：\n"
            "题面：女演员在试镜前剪断了自己的裙子，却最终被录取。\n"
            "答案：这名女演员事先得知试镜剧本中有一幕裙子被撕裂的情节。她故意提前剪开裙子并精心处理切口，使在表演时裙子自然裂开看起来逼真震撼。评审认为她的表演最具冲击力，毫不犹豫录取了她。她的破坏行为反而让她脱颖而出。\n\n"
            "【输出格式】：\n"
            "题面：XXX\n"
            "答案：XXX\n\n"
            f"请基于「{selected_theme}」主题生成一个完全原创的反转推理谜题。"
        )

        return prompt

    async def _generate_for_storage(self) -> bool:
        """为存储库生成故事"""
        try:
            puzzle, answer = await self.generate_story_with_llm()
            if puzzle and answer and not puzzle.startswith("（"):
                self.local_story_storage.add_story(puzzle, answer)
                logger.info("为存储库生成故事成功")
                return True
            else:
                logger.warning("为存储库生成故事失败")
                return False
        except Exception as e:
            logger.error(f"为存储库生成故事错误: {e}")
            return False

    # ✅ 验证用户推理
    async def _get_judge_engine(self, umo: str | None = None) -> JudgeEngine:
        """Resolve configured providers, respecting the current conversation model.

        Args:
            umo: Unified message origin of the current event.

        Returns:
            A stateless judge using the selected providers.

        Raises:
            ValueError: A required provider is unavailable.
        """
        if self.judge_llm_provider_id:
            provider = self.context.get_provider_by_id(self.judge_llm_provider_id)
        else:
            resolver = getattr(self.context, "get_using_provider_async", None)
            if resolver is not None:
                provider = await resolver(umo=umo)
            else:
                provider = self.context.get_using_provider(umo=umo)
        if provider is None:
            raise ValueError("Judgment provider unavailable")
        reviewer = provider
        if self.review_llm_provider_id:
            reviewer = self.context.get_provider_by_id(self.review_llm_provider_id)
            if reviewer is None:
                raise ValueError("Review provider unavailable")
        return JudgeEngine(provider, reviewer)

    async def verify_user_guess(
        self,
        user_guess: str,
        true_answer: str,
        *,
        puzzle: str = "",
        game: dict | None = None,
        umo: str | None = None,
    ) -> VerificationResult:
        """Verify a submitted solution without exposing model-generated evidence.

        Args:
            user_guess: Submitted explanation.
            true_answer: Original solution.
            puzzle: Original puzzle.
            game: Current game state, including public history and private facts.
            umo: Current conversation origin for provider selection.

        Returns:
            A fixed public grade, or a retry result which consumes no attempt.
        """
        state = game or {}
        try:
            engine = await self._get_judge_engine(umo)
            result = await engine.judge(
                mode="verification",
                puzzle=puzzle,
                answer=true_answer,
                content=user_guess,
                history=state.get("qa_history", []),
                hints=state.get("hint_history", []),
                facts=state.get("facts", []),
            )
        except Exception as error:
            logger.warning(
                f"Solution verification unavailable ({type(error).__name__})"
            )
            result = Judgment("error")
        if result.status != "ok":
            return VerificationResult("验证失败", result.response, judgment=result)
        return VerificationResult(
            result.label,
            LEVEL_COMMENTS[result.label],
            result.label in {"完全还原", "核心推理正确"},
            judgment=result,
        )

    async def judge_question(
        self,
        puzzle: str,
        question: str,
        true_answer: str,
        *,
        game: dict | None = None,
        umo: str | None = None,
    ) -> Judgment:
        """Judge a question against source and history using independent review.

        Args:
            puzzle: Original puzzle text.
            question: Player question or statement.
            true_answer: Original solution text.
            game: Current game state.
            umo: Current conversation origin.

        Returns:
            A validated result whose response contains no private evidence.
        """
        state = game or {}
        try:
            engine = await self._get_judge_engine(umo)
            return await engine.judge(
                mode="question",
                puzzle=puzzle,
                answer=true_answer,
                content=question,
                history=state.get("qa_history", []),
                hints=state.get("hint_history", []),
                facts=state.get("facts", []),
            )
        except Exception as error:
            logger.warning(f"Question judgment unavailable ({type(error).__name__})")
            return Judgment("error")

    async def refine_keywords(self, true_answer: str) -> list[str]:
        """使用 LLM 在开局时提炼汤底关键词"""

        # 根据配置获取指定的判断 LLM 提供商
        if self.judge_llm_provider_id:
            provider = self.context.get_provider_by_id(self.judge_llm_provider_id)
            if provider is None:
                logger.error(
                    f"未找到指定的判断 LLM 提供商: {self.judge_llm_provider_id}"
                )
                return []
        else:
            provider = self.context.get_using_provider()
            if provider is None:
                return []

        prompt = (
            f"海龟汤游戏规则：\n"
            f"故事的完整真相是：{true_answer}\n"
            f"任务: 从真相中提炼 8-15 个用于命中判断的关键词或短词。\n"
            f"要求:\n"
            f"1. 关键词应尽量是名词、动词或关键短语，长度 2-8 字；\n"
            f"2. 不要输出整句，不要输出解释；\n"
            f"3. 仅输出关键词，用英文逗号分隔，例如：镜子,墓地,梦游,香水\n"
        )

        try:
            llm_resp: LLMResponse = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                func_tool=None,
                image_urls=[],
                system_prompt="你是一个海龟汤推理游戏的助手。你必须严格按照游戏规则输出内容不能有任何的情感，不能有任何联想。",
            )

            reply = llm_resp.completion_text.strip()
            if not reply:
                return []

            raw_keywords = (
                reply.replace("，", ",")
                .replace("、", ",")
                .replace("\n", ",")
                .split(",")
            )

            keywords: list[str] = []
            for kw in raw_keywords:
                item = kw.strip().strip("。；;:：")
                if len(item) < 2 or len(item) > 12:
                    continue
                if item not in keywords:
                    keywords.append(item)

            return keywords

        except Exception as e:
            logger.error(f"判断问题失败: {e}")
            return []

    # ✅ 生成方向性提示
    def build_allow_list(
        self, puzzle: str, qa_history: list[dict[str, str]]
    ) -> list[str]:
        """根据题面和历史问答构建允许在提示中出现的名词列表"""
        import re

        # 汇总文本：题面 + 所有问答
        parts = [puzzle] + [
            f"{item.get('question', '')}{item.get('answer', '')}" for item in qa_history
        ]
        text = "\n".join(parts)

        # 提取连续的中文、字母或数字片段作为候选名词
        tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text)

        allow: list[str] = []
        for token in tokens:
            if not token:
                continue
            # 同义合并：例如“男人A”“嫌疑人A”只保留末尾的大写字母
            m = re.match(r".*([A-Z])$", token)
            if m:
                token = m.group(1)
            if token not in allow:
                allow.append(token)
        return allow

    async def generate_hint(
        self,
        puzzle: str,
        true_answer: str,
        qa_history: list[dict[str, str]],
        hint_history: list[str],
        allow_list: list[str],
    ) -> str:
        """根据本局已记录的问答与提示生成新的方向性提示"""
        if self.judge_llm_provider_id:
            provider = self.context.get_provider_by_id(self.judge_llm_provider_id)
            if provider is None:
                logger.error(
                    f"未找到指定的判断 LLM 提供商: {self.judge_llm_provider_id}"
                )
                return "（未配置判断 LLM，无法提供提示）"
        else:
            provider = self.context.get_using_provider()
            if provider is None:
                return "（未配置 LLM，无法提供提示）"

        history_text = "\n".join(
            [f"问：{item['question']}\n答：{item['answer']}" for item in qa_history]
        )
        hint_text = "\n".join(hint_history) if hint_history else "（无）"
        allow_text = ", ".join(allow_list) if allow_list else "（无）"
        prompt = (
            '你是"海龟汤"提示生成器。你知道完整真相（仅供内部推理，严禁外泄）。\n'
            "材料：\n\n"
            f"* 题面：{puzzle}\n"
            f"* 完整真相（不可外泄）：{true_answer}\n"
            f"* 历史问答：{history_text}\n"
            f"* 历史提示：{hint_text}\n"
            f"* 允许名词 allow_list：{allow_text}（只能使用其中名词，不得创造新名词）\n\n"
            "在心中完成：\n\n"
            "1. 从历史问答归纳：已确认/已否定/不重要/部分正确的信息；\n"
            "2. 用以下维度整理：对象/身份、关系、动机、时间、地点、证据、步骤、先后、条件、规则、误解；\n"
            "3. 选择一个“未探索”或“partial 尚缺”的维度，且与历史提示不重复；\n"
            "4. 只使用 allow_list 中的名词与通用词，生成一句**动作化**的下一步提问方向；\n"
            "5. 禁止泄露真相细节，不得同义改写泄露；不得复述已确认内容。\n\n"
            "输出要求（只输出一句）：\n\n"
            "* 格式：关注【<维度>】：<动词 + allow_list名词/通用词>\n"
            "* 字数 ≤ 22（或 ≤ 24），不得添加解释。"
        )

        try:
            llm_resp: LLMResponse = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                func_tool=None,
                image_urls=[],
            )
            text = llm_resp.completion_text.strip()
            if text.startswith("提示："):
                text = text[len("提示：") :]
            return text
        except Exception as e:
            logger.error(f"生成提示失败: {e}")
            return "（生成提示失败，请重试）"

    @filter.command("汤难度")
    async def set_difficulty(self, event: AstrMessageEvent, level: str = ""):
        """设置游戏难度"""
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return
        if self.game_state.is_game_active(group_id):
            yield event.plain_result("当前有活跃游戏，无法修改难度")
            return
        if level not in self.difficulty_settings:
            options = "/".join(self.difficulty_settings.keys())
            current = self.group_difficulty.get(group_id, "普通")
            yield event.plain_result(f"可选难度：{options}\n当前难度：{current}")
            return
        self.group_difficulty[group_id] = level
        yield event.plain_result(f"难度已设置为 {level}")

    @filter.command("汤进度开关")
    async def toggle_reveal_progress(self, event: AstrMessageEvent, action: str = ""):
        """切换汤底真相进度功能"""
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return
        if self.game_state.is_game_active(group_id):
            yield event.plain_result("当前有活跃游戏，无法修改设置")
            return

        if action.lower() in ["开", "启用", "on", "true", "1"]:
            self.reveal_progress = True
            yield event.plain_result(
                "✅ 汤底真相进度功能已启用，下一局游戏将显示真相进度"
            )
        elif action.lower() in ["关", "禁用", "off", "false", "0"]:
            self.reveal_progress = False
            yield event.plain_result("❌ 汤底真相进度功能已禁用")
        else:
            status = "已启用" if self.reveal_progress else "已禁用"
            yield event.plain_result(
                f"汤底真相进度功能当前{status}\n使用语法: /汤进度开关 [开|关]"
            )

    # 🎮 开始游戏指令
    @filter.command("汤")
    async def start_soupai_game(self, event: AstrMessageEvent):
        """开始海龟汤游戏

        使用格式: /汤 [题库类型] [题号]

        参数说明:
        - 题库类型 (可选): network(网络题库), storage(本地存储库), custom(自定义题库)
        - 题号 (可选): 指定题库中的题目索引，从0开始

        示例:
        /汤                    # 使用配置的策略随机获取谜题
        /汤 network           # 从网络题库随机获取谜题
        /汤 storage 5         # 从本地存储库获取第5号谜题
        /汤 custom 2          # 从自定义题库获取第2号谜题
        """
        group_id = event.get_group_id()
        logger.info(f"收到开始游戏指令，群ID: {group_id}")

        if not group_id:
            yield event.plain_result("海龟汤游戏只能在群聊中进行哦~")
            return

        # 检查是否已有活跃游戏
        if self.game_state.is_game_active(group_id):
            logger.info(f"群 {group_id} 已有活跃游戏")
            yield event.plain_result(
                "当前群聊已有活跃的海龟汤游戏，请等待游戏结束或使用 /揭晓 结束当前游戏。"
            )
            return

        # 检查是否正在生成谜题
        if group_id in self.generating_games:
            logger.info(f"群 {group_id} 正在生成谜题，忽略重复请求")
            yield event.plain_result("当前有正在生成的谜题，请稍候...")
            return

        self._ensure_story_storages()

        try:
            # 标记正在生成谜题
            self.generating_games.add(group_id)
            logger.info(f"开始为群 {group_id} 生成谜题")

            # 解析命令参数
            message_content = event.message_str.strip()
            args = message_content.split()[1:]  # 去掉命令本身

            story = None
            source_type = None
            puzzle_index = None

            # 解析参数格式: /汤 <network|storage|custom> <题号>
            # 两个参数都是可选的
            if len(args) >= 1:
                first_arg = args[0].lower()

                # 检查第一个参数是否是题库类型
                if first_arg in ["network", "local", "custom"]:
                    source_type = first_arg

                    # 检查是否有第二个参数（题号）
                    if len(args) >= 2:
                        try:
                            puzzle_index = int(args[1])
                        except ValueError:
                            yield event.plain_result("题号必须是数字")
                            self.generating_games.discard(group_id)
                            return
                else:
                    # 第一个参数不是题库类型，可能是题号
                    try:
                        puzzle_index = int(first_arg)
                        # 使用配置的策略作为默认题库类型
                        strategy = self.puzzle_source_strategy
                        if strategy == "network_first":
                            source_type = "network"
                        elif strategy == "local_first":
                            source_type = "local"
                        elif strategy == "custom_first":
                            source_type = "custom"
                        else:  # random
                            source_type = "current"
                    except ValueError:
                        # 第一个参数既不是题库类型也不是题号，使用默认策略随机获取
                        source_type = "current"
            else:
                # 没有参数，使用配置的策略随机获取
                source_type = "current"

            # 根据解析的参数获取故事
            if puzzle_index is not None:
                # 指定了题号，从特定题库获取
                story = await self.get_story_by_index(source_type, puzzle_index)
                if not story:
                    yield event.plain_result(
                        f"{source_type}题库中没有第 {puzzle_index} 号题目"
                    )
                    self.generating_games.discard(group_id)
                    return
            else:
                # 没有指定题号，根据策略随机获取
                if source_type == "current":
                    # 使用配置的策略获取随机故事
                    strategy = self.puzzle_source_strategy
                    story = await self.get_story_by_strategy(strategy)
                else:
                    # 从指定题库获取随机故事
                    if source_type == "network":
                        story = self.online_story_storage.get_story()
                    elif source_type == "local":
                        story = self.local_story_storage.get_story()
                    elif source_type == "custom":
                        story = self.custom_story_storage.get_story()
                    else:
                        yield event.plain_result(
                            "题库类型参数错误，请使用 network/local/custom"
                        )
                        self.generating_games.discard(group_id)
                        return

            if not story:
                yield event.plain_result("获取谜题失败，请重试")
                self.generating_games.discard(group_id)
                return

            puzzle, answer = story

            # 检查LLM生成是否失败
            if puzzle == "（无法生成题面，请先配置大语言模型）":
                yield event.plain_result(f"生成谜题失败：{answer}")
                self.generating_games.discard(group_id)
                return

            difficulty = self.group_difficulty.get(group_id, "普通")
            diff_conf = self.difficulty_settings.get(
                difficulty, self.difficulty_settings["普通"]
            )

            # 初始化知道的答案（如果启用真相进度功能）
            knowed_answer = (
                self.mask_answer_with_progress(answer) if self.reveal_progress else None
            )
            answer_keywords: list[str] = []
            if self.reveal_progress:
                answer_keywords = await self.refine_keywords(answer)

            facts = []
            try:
                engine = await self._get_judge_engine(event.unified_msg_origin)
                facts = await engine.prepare_facts(puzzle, answer)
            except Exception as error:
                logger.warning(f"Fact index unavailable ({type(error).__name__})")

            if self.game_state.start_game(
                group_id,
                puzzle,
                answer,
                difficulty=difficulty,
                question_limit=diff_conf["limit"],
                question_count=0,
                verification_attempts=0,
                early_verification_attempts=0,
                facts=facts,
                accept_levels=diff_conf["accept_levels"],
                hint_limit=diff_conf.get("hint_limit"),
                hint_count=0,
                knowed_answer=knowed_answer,
                answer_keywords=answer_keywords,
            ):
                extra = ""
                if diff_conf["limit"] is not None:
                    extra = f"\n模式：{difficulty}（{diff_conf['limit']} 次提问"
                else:
                    extra = f"\n模式：{difficulty}（无限提问"

                hint_limit = diff_conf.get("hint_limit")
                if hint_limit == 0:
                    extra += "，无提示）"
                elif hint_limit is not None:
                    extra += f"，{hint_limit} 次提示）"
                else:
                    extra += "）"

                # 构建游戏开始消息
                game_msg = f"🎮 海龟汤游戏开始！{extra}\n\n📖 题面：{puzzle}\n\n💡 请直接提问或陈述，我会回答：是、否、是或不是、不重要\n💡 输入 /提示 可以获取方向性提示\n💡 输入 /验证 <答案> 可以验证答案是否正确\n💡 输入 /揭晓 可以查看完整故事"

                # 如果启用了真相进度，在游戏开始时显示
                if self.reveal_progress and knowed_answer:
                    game_msg += f"\n\n📍 汤底真相进度如下：\n{knowed_answer}"

                yield event.plain_result(game_msg)

                # 启动会话控制
                await self._start_game_session(event, group_id, answer)
            else:
                yield event.plain_result("游戏启动失败，请重试")

            # 移除生成状态，因为故事已经准备完成
            self.generating_games.discard(group_id)
            logger.info(f"群 {group_id} 故事准备完成，移除生成状态")

        except Exception as e:
            logger.error(f"启动游戏失败: {e}")
            # 发生异常时也要移除生成状态
            self.generating_games.discard(group_id)
            logger.info(f"群 {group_id} 启动游戏异常，移除生成状态")
            yield event.plain_result(f"启动游戏时发生错误：{e}")

    # 🔍 揭晓指令
    @filter.command("揭晓")
    async def reveal_answer(self, event: AstrMessageEvent):
        """揭晓答案"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("海龟汤游戏只能在群聊中进行哦~")
            return

        # 检查是否有活跃游戏，如果有活跃游戏，说明在会话控制中，不在这里处理
        if self.game_state.is_game_active(group_id):
            # 阻止事件继续传播，避免被会话控制系统重复处理
            await event.block()
            return
        game = self.game_state.get_game(group_id)
        if not game:
            yield event.plain_result(
                "当前没有活跃的海龟汤游戏，请使用 /汤 开始新游戏。"
            )
            return

        answer = game["answer"]
        puzzle = game["puzzle"]

        # 发送完整的揭晓信息
        yield event.plain_result(
            f"🎯 海龟汤游戏结束！\n\n📖 题面：{puzzle}\n📖 完整故事：{answer}\n\n感谢参与游戏！"
        )

        # 结束游戏
        self.game_state.end_game(group_id)
        logger.info(f"游戏已结束，群ID: {group_id}")

    async def _handle_question_in_session(
        self,
        event: AstrMessageEvent,
        group_id: str,
        controller: SessionController,
        question: str,
    ) -> None:
        """Commit only accepted judgments for the same still-active game.

        Args:
            event: Incoming player event.
            group_id: Current group identifier.
            controller: Session timeout controller.
            question: Player question stripped of surrounding whitespace.
        """
        game = self.game_state.get_game(group_id)
        if game is None:
            return
        limit = game.get("question_limit")
        if limit is not None and game.get("question_count", 0) >= limit:
            return
        controller.keep(timeout=self.game_timeout, reset_timeout=True)
        result = await self.judge_question(
            game["puzzle"],
            question,
            game["answer"],
            game=game,
            umo=event.unified_msg_origin,
        )
        if self.game_state.get_game(group_id) is not game:
            return
        if result.status != "ok":
            await event.send(event.plain_result(result.response))
            controller.keep(timeout=self.game_timeout, reset_timeout=True)
            return

        history = game.setdefault("qa_history", [])
        notices = []
        for correction in result.corrections:
            item = history[correction["index"] - 1]
            if item["answer"] != correction["answer"]:
                item.setdefault("original_answer", item["answer"])
                item["answer"] = correction["answer"]
                notices.append(
                    f"更正第{correction['index']}问：{correction['answer']}。"
                )
        history.append({"question": question, "answer": result.label})
        game["question_count"] = game.get("question_count", 0) + 1
        if result.label == "是" and self.reveal_progress and game.get("knowed_answer"):
            previous = game["knowed_answer"]
            for keyword in game.get("answer_keywords", []):
                if keyword and keyword in question:
                    game["knowed_answer"] = self._merge_reveal_progress(
                        game["knowed_answer"],
                        keyword,
                        game["answer"],
                    )
            if previous != game["knowed_answer"]:
                notices.append("真相进度已更新。")
        reply = result.label
        if limit is not None:
            reply += f"（{game['question_count']}/{limit}）"
            if game["question_count"] >= limit:
                reply += "\n提问次数已用完，你有2次验证机会，请使用 /验证 <推理内容>。"
        await event.send(event.plain_result("\n".join([*notices, reply])))
        if self.game_state.get_game(group_id) is game:
            controller.keep(timeout=self.game_timeout, reset_timeout=True)

    # 🎯 游戏会话控制
    async def _start_game_session(
        self, event: AstrMessageEvent, group_id: str, answer: str
    ):
        """启动游戏会话控制"""
        task = asyncio.current_task()
        if task is not None:
            self._session_tasks.add(task)
        try:

            @session_waiter(timeout=self.game_timeout, record_history_chains=False)
            async def game_session_waiter(
                controller: SessionController, event: AstrMessageEvent
            ):
                try:
                    # 从游戏状态获取答案，确保变量可用
                    game = self.game_state.get_game(group_id)
                    if not game:
                        return
                    current_answer = game["answer"]
                    user_input = event.message_str.strip()
                    logger.info(f"会话控制收到消息: '{user_input}'")

                    # 允许在会话中使用 /汤状态 和 /强制结束 指令
                    if user_input in ("/汤状态", "汤状态"):
                        await self._handle_game_status_in_session(event, group_id)
                        return

                    if user_input in ("/强制结束", "强制结束"):
                        await self._handle_force_end_in_session(event, group_id)
                        if not self.game_state.is_game_active(group_id):
                            controller.stop()
                        return

                    normalized_input = user_input.lstrip("/").strip()
                    if normalized_input == "查看":
                        await self._handle_view_history_in_session(event, group_id)
                        controller.keep(timeout=self.game_timeout, reset_timeout=True)
                        return
                    if user_input in ("/提示", "提示"):
                        async for result in self.hint_command(event):
                            await event.send(result)
                        controller.keep(timeout=self.game_timeout, reset_timeout=True)
                        return
                    # 特殊处理 /验证 指令
                    if user_input.startswith("/验证"):
                        import re

                        match = re.match(r"^/验证\s*(.+)$", user_input)
                        if match:
                            user_guess = match.group(1).strip()
                            # 手动调用验证函数
                            await self._handle_verification_in_session(
                                event, user_guess, current_answer
                            )
                            # 检查游戏是否已结束（用户可能猜中了）
                            if not self.game_state.is_game_active(group_id):
                                controller.stop()
                                return
                        else:
                            await event.send(
                                event.plain_result(
                                    "请输入要验证的内容，例如：/验证 他是她的父亲"
                                )
                            )
                        return
                    elif user_input.startswith("验证"):
                        import re

                        match = re.match(r"^验证\s*(.+)$", user_input)
                        if match:
                            user_guess = match.group(1).strip()
                            # 手动调用验证函数
                            await self._handle_verification_in_session(
                                event, user_guess, current_answer
                            )
                            # 检查游戏是否已结束（用户可能猜中了）
                            if not self.game_state.is_game_active(group_id):
                                controller.stop()
                                return
                        else:
                            await event.send(
                                event.plain_result(
                                    "请输入要验证的内容，例如：验证 他是她的父亲"
                                )
                            )
                        return
                    # 特殊处理 /揭晓 指令
                    if user_input == "揭晓":
                        # 获取游戏信息并发送答案
                        game = self.game_state.get_game(group_id)
                        if game:
                            answer = game["answer"]
                            puzzle = game["puzzle"]
                            await event.send(
                                event.plain_result(
                                    f"🎯 海龟汤游戏结束！\n\n📖 题面：{puzzle}\n📖 完整故事：{answer}\n\n感谢参与游戏！"
                                )
                            )
                            self.game_state.end_game(group_id)
                        controller.stop()
                        return

                    # 特殊处理 /查看进度 指令（查看汤底真相进度）
                    if user_input in ("/查看进度", "查看进度"):
                        game = self.game_state.get_game(group_id)
                        if game and game.get("knowed_answer"):
                            await event.send(
                                event.plain_result(
                                    f"📍 汤底真相进度如下：\n{game['knowed_answer']}"
                                )
                            )
                        else:
                            await event.send(
                                event.plain_result("汤底真相进度功能未启用或暂无进度")
                            )
                        controller.keep(timeout=self.game_timeout, reset_timeout=True)
                        return

                    # Step 1: 检查是否是 /开头的命令，如果是则忽略，让指令处理器处理
                    if user_input.startswith("/"):
                        # 不处理指令，让事件继续传播到指令处理器
                        return
                    # Step 2: 检查是否 @了 bot，只有@bot的消息才触发问答判断
                    if not self._is_at_bot(event):
                        return
                    # Step 3: 是@bot的自然语言提问，触发 LLM 判断
                    game = self.game_state.get_game(group_id)
                    question_limit = game.get("question_limit") if game else None
                    question_count = game.get("question_count", 0) if game else 0
                    if question_limit is not None and question_count >= question_limit:
                        remaining = 2 - game.get("verification_attempts", 0)
                        await event.send(
                            event.plain_result(
                                f"❗️提问次数已用完，请使用 /验证 进行猜测（剩余{remaining}次验证机会）"
                            )
                        )
                        return

                    await self._handle_question_in_session(
                        event, group_id, controller, user_input.strip()
                    )

                except Exception as e:
                    logger.error(f"会话控制内部错误: {e}")
                    await event.send(event.plain_result(f"游戏处理过程中发生错误：{e}"))
                    # 如果发生错误，结束游戏
                    self.game_state.end_game(group_id)
                    controller.stop()

            try:
                # 使用群 ID 限制会话范围，避免多个群并发时互相触发
                await game_session_waiter(
                    event, session_filter=GroupSessionFilter(group_id)
                )
            except TimeoutError:
                game = self.game_state.get_game(group_id)
                if game:
                    await event.send(
                        event.plain_result(
                            f"⏰ 游戏超时！\n\n📖 完整故事：{game['answer']}\n\n游戏结束！"
                        )
                    )
                    self.game_state.end_game(group_id)
            except Exception as e:
                logger.error(f"游戏会话错误: {e}")
                await event.send(event.plain_result(f"游戏过程中发生错误：{e}"))
                self.game_state.end_game(group_id)
        except Exception as e:
            logger.error(f"启动游戏会话失败: {e}")
            await event.send(event.plain_result(f"启动游戏会话失败：{e}"))
        finally:
            self._session_tasks.discard(task)

    def _is_at_bot(self, event: AstrMessageEvent) -> bool:
        """检查消息是否@了bot"""

        bot_id = str(event.get_self_id())
        for comp in event.message_obj.message:
            if isinstance(comp, At) and str(comp.qq) == bot_id:
                return True
        return False

    async def get_story_by_strategy(self, strategy: str) -> tuple[str, str] | None:
        """根据策略获取故事，返回 (puzzle, answer) 或 None"""
        import random

        self._ensure_story_storages()

        if strategy == "network_first":
            # 策略1：优先网络题库 -> 本地存储库 -> 自定义题库 -> LLM现场生成

            # 1. 检查网络题库
            story = self.online_story_storage.get_story()
            if story:
                return story

            # 2. 检查本地存储库
            story = self.local_story_storage.get_story()
            if story:
                return story

            # 3. 检查自定义题库
            story = self.custom_story_storage.get_story()
            if story:
                return story

            # 4. LLM现场生成
            return await self.generate_story_with_llm()

        elif strategy == "local_first":
            # 策略2：优先本地存储库 -> 网络题库 -> 自定义题库 -> LLM现场生成

            # 1. 检查本地存储库
            story = self.local_story_storage.get_story()
            if story:
                return story

            # 2. 检查网络题库
            story = self.online_story_storage.get_story()
            if story:
                return story

            # 3. 检查自定义题库
            story = self.custom_story_storage.get_story()
            if story:
                return story

            # 4. LLM现场生成
            return await self.generate_story_with_llm()

        elif strategy == "custom_first":
            # 策略3：优先自定义题库 -> 本地存储库 -> 网络题库 -> LLM现场生成

            # 1. 检查自定义题库
            story = self.custom_story_storage.get_story()
            if story:
                return story

            # 2. 检查本地存储库
            story = self.local_story_storage.get_story()
            if story:
                return story

            # 3. 检查网络题库
            story = self.online_story_storage.get_story()
            if story:
                return story

            # 4. LLM现场生成
            return await self.generate_story_with_llm()

        elif strategy == "random":
            # 策略3：随机选择网络题库、本地存储库或自定义题库，失败时使用LLM现场生成

            # 随机决定这次从哪个题库获取
            choice = random.choice(["network", "local", "custom"])
            if choice == "network":
                # 参考策略1的网络题库逻辑
                story = self.online_story_storage.get_story()
                if story:
                    return story

                story = self.local_story_storage.get_story()
                if story:
                    return story

                story = self.custom_story_storage.get_story()
                if story:
                    return story

                return await self.generate_story_with_llm()
            elif choice == "local":
                # 参考策略2的本地存储库逻辑
                story = self.local_story_storage.get_story()
                if story:
                    return story

                story = self.online_story_storage.get_story()
                if story:
                    return story

                story = self.custom_story_storage.get_story()
                if story:
                    return story

                return await self.generate_story_with_llm()
            else:  # custom
                # 优先自定义题库
                story = self.custom_story_storage.get_story()
                if story:
                    return story

                story = self.local_story_storage.get_story()
                if story:
                    return story

                story = self.online_story_storage.get_story()
                if story:
                    return story

                return await self.generate_story_with_llm()

        return None

    async def get_story_by_index(
        self, source_type: str, index: int
    ) -> tuple[str, str] | None:
        """根据索引获取特定故事

        Args:
            source_type: "network" - 网络题库, "current" - 当前策略题库, "custom" - 自定义题库
            index: 题目索引（从0开始）

        Returns:
            (puzzle, answer) 或 None
        """
        self._ensure_story_storages()

        if source_type == "network":
            # 从网络题库获取指定索引的故事
            if index < 0 or index >= len(self.online_story_storage.stories):
                return None

            story = self.online_story_storage.stories[index]
            # 标记为已使用
            with self.online_story_storage.lock:
                self.online_story_storage.used_indexes.add(index)
                self.online_story_storage.save_usage_record()

            logger.info(f"从网络题库获取指定故事，索引: {index}")
            return story["puzzle"], story["answer"]

        elif source_type == "custom":
            # 从自定义题库获取指定索引的故事
            if index < 0 or index >= len(self.custom_story_storage.stories):
                return None

            story = self.custom_story_storage.stories[index]
            # 标记为已使用
            with self.custom_story_storage.lock:
                self.custom_story_storage.used_indexes.add(index)
                self.custom_story_storage.save_usage_record()

            logger.info(f"从自定义题库获取指定故事，索引: {index}")
            return story["puzzle"], story["answer"]

        elif source_type == "current":
            # 根据当前策略获取指定索引的故事
            strategy = self.puzzle_source_strategy

            if strategy == "network_first":
                # 优先检查网络题库
                if index < len(self.online_story_storage.stories):
                    story = self.online_story_storage.stories[index]
                    with self.online_story_storage.lock:
                        self.online_story_storage.used_indexes.add(index)
                        self.online_story_storage.save_usage_record()
                    logger.info(f"从网络题库获取指定故事，索引: {index}")
                    return story["puzzle"], story["answer"]

                # 然后检查本地存储库
                local_index = index - len(self.online_story_storage.stories)
                if local_index >= 0 and local_index < len(
                    self.local_story_storage.stories
                ):
                    story = self.local_story_storage.stories[local_index]
                    with self.local_story_storage.lock:
                        self.local_story_storage.used_indexes.add(local_index)
                        self.local_story_storage.save_usage_record()
                    logger.info(f"从本地存储库获取指定故事，索引: {local_index}")
                    return story["puzzle"], story["answer"]

                # 然后检查自定义题库
                custom_index = local_index - len(self.local_story_storage.stories)
                if custom_index >= 0 and custom_index < len(
                    self.custom_story_storage.stories
                ):
                    story = self.custom_story_storage.stories[custom_index]
                    with self.custom_story_storage.lock:
                        self.custom_story_storage.used_indexes.add(custom_index)
                        self.custom_story_storage.save_usage_record()
                    logger.info(f"从自定义题库获取指定故事，索引: {custom_index}")
                    return story["puzzle"], story["answer"]

                # 超出范围，返回None
                return None

            elif strategy == "local_first":
                # 优先检查本地存储库
                if index < len(self.local_story_storage.stories):
                    story = self.local_story_storage.stories[index]
                    with self.local_story_storage.lock:
                        self.local_story_storage.used_indexes.add(index)
                        self.local_story_storage.save_usage_record()
                    logger.info(f"从本地存储库获取指定故事，索引: {index}")
                    return story["puzzle"], story["answer"]

                # 然后检查网络题库
                network_index = index - len(self.local_story_storage.stories)
                if network_index >= 0 and network_index < len(
                    self.online_story_storage.stories
                ):
                    story = self.online_story_storage.stories[network_index]
                    with self.online_story_storage.lock:
                        self.online_story_storage.used_indexes.add(network_index)
                        self.online_story_storage.save_usage_record()
                    logger.info(f"从网络题库获取指定故事，索引: {network_index}")
                    return story["puzzle"], story["answer"]

                # 然后检查自定义题库
                custom_index = network_index - len(self.online_story_storage.stories)
                if custom_index >= 0 and custom_index < len(
                    self.custom_story_storage.stories
                ):
                    story = self.custom_story_storage.stories[custom_index]
                    with self.custom_story_storage.lock:
                        self.custom_story_storage.used_indexes.add(custom_index)
                        self.custom_story_storage.save_usage_record()
                    logger.info(f"从自定义题库获取指定故事，索引: {custom_index}")
                    return story["puzzle"], story["answer"]

                # 超出范围，返回None
                return None

            elif strategy == "custom_first":
                # 优先检查自定义题库
                if index < len(self.custom_story_storage.stories):
                    story = self.custom_story_storage.stories[index]
                    with self.custom_story_storage.lock:
                        self.custom_story_storage.used_indexes.add(index)
                        self.custom_story_storage.save_usage_record()
                    logger.info(f"从自定义题库获取指定故事，索引: {index}")
                    return story["puzzle"], story["answer"]

                # 然后检查本地存储库
                local_index = index - len(self.custom_story_storage.stories)
                if local_index >= 0 and local_index < len(
                    self.local_story_storage.stories
                ):
                    story = self.local_story_storage.stories[local_index]
                    with self.local_story_storage.lock:
                        self.local_story_storage.used_indexes.add(local_index)
                        self.local_story_storage.save_usage_record()
                    logger.info(f"从本地存储库获取指定故事，索引: {local_index}")
                    return story["puzzle"], story["answer"]

                # 然后检查网络题库
                network_index = local_index - len(self.local_story_storage.stories)
                if network_index >= 0 and network_index < len(
                    self.online_story_storage.stories
                ):
                    story = self.online_story_storage.stories[network_index]
                    with self.online_story_storage.lock:
                        self.online_story_storage.used_indexes.add(network_index)
                        self.online_story_storage.save_usage_record()
                    logger.info(f"从网络题库获取指定故事，索引: {network_index}")
                    return story["puzzle"], story["answer"]

                # 超出范围，返回None
                return None

            elif strategy == "random":
                # 对于随机策略，我们无法准确知道索引对应哪个题库
                # 这里我们按顺序检查：先网络题库，再本地存储库，最后自定义题库
                if index < len(self.online_story_storage.stories):
                    story = self.online_story_storage.stories[index]
                    with self.online_story_storage.lock:
                        self.online_story_storage.used_indexes.add(index)
                        self.online_story_storage.save_usage_record()
                    logger.info(f"从网络题库获取指定故事，索引: {index}")
                    return story["puzzle"], story["answer"]

                local_index = index - len(self.online_story_storage.stories)
                if local_index >= 0 and local_index < len(
                    self.local_story_storage.stories
                ):
                    story = self.local_story_storage.stories[local_index]
                    with self.local_story_storage.lock:
                        self.local_story_storage.used_indexes.add(local_index)
                        self.local_story_storage.save_usage_record()
                    logger.info(f"从本地存储库获取指定故事，索引: {local_index}")
                    return story["puzzle"], story["answer"]

                custom_index = local_index - len(self.local_story_storage.stories)
                if custom_index >= 0 and custom_index < len(
                    self.custom_story_storage.stories
                ):
                    story = self.custom_story_storage.stories[custom_index]
                    with self.custom_story_storage.lock:
                        self.custom_story_storage.used_indexes.add(custom_index)
                        self.custom_story_storage.save_usage_record()
                    logger.info(f"从自定义题库获取指定故事，索引: {custom_index}")
                    return story["puzzle"], story["answer"]

                # 超出范围，返回None
                return None

        return None

    async def _handle_game_status_in_session(
        self, event: AstrMessageEvent, group_id: str
    ):
        """在会话控制中处理游戏状态查询逻辑"""
        try:
            if self.game_state.is_game_active(group_id):
                game = self.game_state.get_game(group_id)
                difficulty = game.get("difficulty", "普通")
                question_count = game.get("question_count", 0)
                question_limit = game.get("question_limit")
                hint_count = game.get("hint_count", 0)
                hint_limit = game.get("hint_limit")

                question_info = (
                    f"{question_count}/{question_limit}"
                    if question_limit
                    else f"{question_count}/∞"
                )
                hint_info = f"{hint_count}/{hint_limit}" if hint_limit else "不可用"

                await event.send(
                    event.plain_result(
                        f"🎮 当前有活跃的海龟汤游戏\n📖 题面：{game['puzzle']}\n🎯 难度：{difficulty}\n❓ 提问：{question_info}\n💡 提示：{hint_info}"
                    )
                )
            else:
                await event.send(
                    event.plain_result(
                        "🎮 当前没有活跃的海龟汤游戏\n💡 使用 /汤 开始新游戏"
                    )
                )

        except Exception as e:
            logger.error(f"会话游戏状态查询失败: {e}")
            await event.send(event.plain_result(f"查询游戏状态时发生错误：{e}"))

    async def _handle_force_end_in_session(
        self, event: AstrMessageEvent, group_id: str
    ):
        """在会话控制中处理强制结束游戏逻辑"""
        try:
            if self.game_state.end_game(group_id):
                await event.send(event.plain_result("✅ 已强制结束当前海龟汤游戏"))
            else:
                await event.send(event.plain_result("❌ 当前没有活跃的游戏需要结束"))

        except Exception as e:
            logger.error(f"会话强制结束失败: {e}")
            await event.send(event.plain_result(f"强制结束游戏时发生错误：{e}"))

    async def _handle_view_history_in_session(
        self, event: AstrMessageEvent, group_id: str
    ):
        """在会话控制中处理查看历史记录逻辑"""
        try:
            game = self.game_state.get_game(group_id)
            if not game:
                await event.send(event.plain_result("无法获取游戏状态"))
                return

            history = game.get("qa_history", [])

            if not history:
                await event.send(event.plain_result("目前还没有人提问哦~"))
                return

            lines = ["📋 提问记录："]
            for idx, item in enumerate(history, 1):
                lines.append(f"{idx}. 问：{item['question']}\n   答：{item['answer']}")

            response = "\n".join(lines)
            await event.send(event.plain_result(response))

        except Exception as e:
            logger.error(f"会话查看历史失败: {e}")
            await event.send(event.plain_result(f"查看历史记录时发生错误：{e}"))

    async def _build_hint_result(
        self, event: AstrMessageEvent, group_id: str
    ) -> MessageEventResult | None:
        """生成提示结果，供指令或会话控制调用"""
        if not group_id:
            return event.plain_result("提示功能只能在群聊中使用")

        game = self.game_state.get_game(group_id)
        if not game:
            return event.plain_result("当前没有活跃的海龟汤游戏")

        hint_limit = game.get("hint_limit")
        hint_count = game.get("hint_count", 0)
        if hint_limit == 0:
            return event.plain_result("当前难度不可使用提示")
        if hint_limit is not None and hint_count >= hint_limit:
            return event.plain_result("提示次数已用完")

        qa_history = game.get("qa_history", [])
        if not qa_history:
            return event.plain_result("请先进行提问后再请求提示")

        hint_history = game.get("hint_history", [])
        allow_list = self.build_allow_list(game["puzzle"], qa_history)

        hint = await self.generate_hint(
            game["puzzle"], game["answer"], qa_history, hint_history, allow_list
        )
        game["hint_count"] = hint_count + 1
        game["hint_history"] = hint_history + [hint]
        suffix = ""
        if hint_limit is not None:
            suffix = f"（{game['hint_count']}/{hint_limit}）"
        return event.plain_result(f"提示：{hint}{suffix}")

    async def _handle_verification_in_session(
        self,
        event: AstrMessageEvent,
        user_guess: str,
        answer: str,
    ) -> None:
        """Verify and consume an attempt only after a valid, current result.

        Args:
            event: Current player event.
            user_guess: Submitted complete explanation.
            answer: Original solution of the current game.
        """
        group_id = event.get_group_id()
        game = self.game_state.get_game(group_id) if group_id else None
        if game is None:
            return
        limit = game.get("question_limit")
        exhausted = limit is not None and game.get("question_count", 0) >= limit
        counter = (
            "verification_attempts" if exhausted else "early_verification_attempts"
        )
        limited = game.get("difficulty", "普通") != "简单"
        if limited and game.get(counter, 0) >= 2:
            await event.send(event.plain_result("已经没有验证次数了。"))
            return
        result = await self.verify_user_guess(
            user_guess,
            answer,
            puzzle=game["puzzle"],
            game=game,
            umo=event.unified_msg_origin,
        )
        if self.game_state.get_game(group_id) is not game:
            return
        judgment = result.judgment
        if judgment is None or judgment.status != "ok":
            await event.send(event.plain_result(result.comment))
            return
        notices = []
        for correction in judgment.corrections:
            item = game["qa_history"][correction["index"] - 1]
            if item["answer"] != correction["answer"]:
                item.setdefault("original_answer", item["answer"])
                item["answer"] = correction["answer"]
                notices.append(
                    f"更正第{correction['index']}问：{correction['answer']}。"
                )
        if limited:
            game[counter] = game.get(counter, 0) + 1
        response = "\n".join(
            [*notices, f"等级：{result.level}\n评价：{result.comment}"]
        )
        if result.level in game.get("accept_levels", ["完全还原"]):
            self.game_state.end_game(group_id)
            response += (
                f"\n\n🎉 恭喜！你猜中了！\n\n📖 完整故事：{answer}\n\n游戏结束！"
            )
        elif limited:
            remaining = 2 - game[counter]
            if exhausted and remaining == 0:
                self.game_state.end_game(group_id)
                response += (
                    f"\n\n❌ 验证未通过。\n\n📖 完整故事：{answer}\n\n游戏结束！"
                )
            else:
                response += f"\n\n验证未通过，本阶段剩余 {remaining} 次验证机会。"
        await event.send(event.plain_result(response))

    # 📊 游戏状态查询
    @filter.command("汤状态")
    async def check_game_status(self, event: AstrMessageEvent):
        """查看当前游戏状态"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return

        if self.game_state.is_game_active(group_id):
            game = self.game_state.get_game(group_id)
            difficulty = game.get("difficulty", "普通")
            question_count = game.get("question_count", 0)
            question_limit = game.get("question_limit")
            hint_count = game.get("hint_count", 0)
            hint_limit = game.get("hint_limit")

            question_info = (
                f"{question_count}/{question_limit}"
                if question_limit
                else f"{question_count}/∞"
            )
            hint_info = f"{hint_count}/{hint_limit}" if hint_limit else "不可用"

            yield event.plain_result(
                f"🎮 当前有活跃的海龟汤游戏\n📖 题面：{game['puzzle']}\n🎯 难度：{difficulty}\n❓ 提问：{question_info}\n💡 提示：{hint_info}"
            )
        else:
            yield event.plain_result(
                "🎮 当前没有活跃的海龟汤游戏\n💡 使用 /汤 开始新游戏"
            )

    @filter.command("查看")
    async def view_question_history(self, event: AstrMessageEvent):
        """查看当前已提问的问题及回答"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return
        if not self.game_state.is_game_active(group_id):
            yield event.plain_result("当前没有活跃的海龟汤游戏")
            return
        game = self.game_state.get_game(group_id)
        history = game.get("qa_history", []) if game else []
        if not history:
            yield event.plain_result("目前还没有人提问哦~")
            return
        lines = ["📋 提问记录："]
        for idx, item in enumerate(history, 1):
            lines.append(f"{idx}. 问：{item['question']}\n   答：{item['answer']}")
        yield event.plain_result("\n".join(lines))

    # 🆘 强制结束游戏（管理员功能）
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("强制结束")
    async def force_end_game(self, event: AstrMessageEvent):
        """强制结束当前游戏（仅管理员）"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return

        if self.game_state.end_game(group_id):
            yield event.plain_result("✅ 已强制结束当前海龟汤游戏")
        else:
            yield event.plain_result("❌ 当前没有活跃的游戏需要结束")

    # 📚 备用故事管理指令
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("备用开始")
    async def start_backup_generation(self, event: AstrMessageEvent):
        """开始生成备用故事（仅管理员）"""

        if self.auto_generating:
            yield event.plain_result("⚠️ 备用故事生成已在运行中")
            return

        # 检查存储库是否已满
        self._ensure_story_storages()
        storage_info = self.local_story_storage.get_storage_info()
        if storage_info["available"] <= 0:
            yield event.plain_result("⚠️ 存储库已满，无法生成更多故事")
            return

        self.auto_generating = True
        asyncio.create_task(self._auto_generate_loop())
        yield event.plain_result(
            f"✅ 开始生成备用故事，存储库状态: {storage_info['total']}/{storage_info['max_size']}"
        )

    # 🔒 全局指令拦截器 - 当正在生成时提醒用户
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def global_command_interceptor(self, event: AstrMessageEvent):
        """全局指令拦截器，当正在生成备用故事时提醒用户"""
        # 检查是否有活跃游戏，如果有活跃游戏，不在这里处理
        group_id = event.get_group_id()
        if group_id and self.game_state.is_game_active(group_id):
            # 有活跃游戏，让会话控制处理
            return

        # 如果正在生成备用故事，且不是 /备用结束 指令，则提醒用户
        if self.auto_generating:
            user_input = event.message_str.strip()
            # 只拦截非本插件的指令，避免阻断自己的指令
            if (
                user_input.startswith("/")
                and not user_input.startswith("/备用结束")
                and not user_input.startswith("/汤")
                and not user_input.startswith("/揭晓")
                and not user_input.startswith("/验证")
                and not user_input.startswith("/汤状态")
                and not user_input.startswith("/强制结束")
                and not user_input.startswith("/备用开始")
                and not user_input.startswith("/备用状态")
                and not user_input.startswith("/汤配置")
                and not user_input.startswith("/重置题库")
                and not user_input.startswith("/题库详情")
                and not user_input.startswith("/查看")
                and not user_input.startswith("/提示")
            ):
                yield event.plain_result(
                    "⚠️ 系统正在生成备用故事，请稍后再试或使用 /备用结束 停止生成"
                )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("备用结束")
    async def stop_backup_generation(self, event: AstrMessageEvent):
        """停止生成备用故事（仅管理员）"""

        if not self.auto_generating:
            yield event.plain_result("⚠️ 备用故事生成未在运行")
            return

        self.auto_generating = False
        yield event.plain_result("✅ 已停止生成备用故事，正在完成当前生成...")

    @filter.command("备用状态")
    async def check_backup_status(self, event: AstrMessageEvent):
        """查看备用故事状态"""
        self._ensure_story_storages()
        storage_info = self.local_story_storage.get_storage_info()
        online_info = self.online_story_storage.get_storage_info()
        status = "🟢 运行中" if self.auto_generating else "🔴 已停止"

        # 检查存储库是否已满
        storage_full_warning = ""
        if storage_info["available"] <= 0:
            storage_full_warning = "\n⚠️ 本地存储库已满，自动生成已停止"

        message = (
            f"📚 备用故事状态：\n"
            f"• 生成状态：{status}\n"
            f"• 本地存储库：{storage_info['total']}/{storage_info['max_size']}\n"
            f"• 已使用题目：{storage_info['used']}\n"
            f"• 剩余题目：{storage_info['remaining']}\n"
            f"• 可用空间：{storage_info['available']}\n"
            f"• 网络题库：{online_info['total']} 个 (已用: {online_info['used']}, 剩余: {online_info['available']})\n"
            f"• 自动生成时间：{self.auto_generate_start}:00-{self.auto_generate_end}:00{storage_full_warning}"
        )

        yield event.plain_result(message)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("重置题库")
    async def reset_story_storage(self, event: AstrMessageEvent):
        """重置题库使用记录（仅管理员）"""

        self._ensure_story_storages()

        # 重置网络题库使用记录
        self.online_story_storage.reset_usage()
        online_info = self.online_story_storage.get_storage_info()

        # 重置本地存储库使用记录
        self.local_story_storage.reset_usage()
        local_info = self.local_story_storage.get_storage_info()

        message = (
            f"✅ 题库使用记录已重置！\n"
            f"• 网络题库：{online_info['total']} 个谜题 (已重置)\n"
            f"• 本地存储库：{local_info['total']} 个谜题 (已重置)\n"
            f"• 所有题目现在都可以重新使用"
        )

        yield event.plain_result(message)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("题库详情")
    async def show_storage_details(self, event: AstrMessageEvent):
        """查看题库详细使用记录（仅管理员）"""

        # 确保题库已初始化
        self._ensure_story_storages()

        # 获取网络题库详细信息
        online_info = self.online_story_storage.get_storage_info()
        online_usage = self.online_story_storage.get_usage_info()

        # 获取本地存储库详细信息
        local_info = self.local_story_storage.get_storage_info()
        local_usage = self.local_story_storage.get_usage_info()

        # 安全计算使用率，避免除零错误
        online_usage_rate = (
            (online_info["used"] / online_info["total"] * 100)
            if online_info["total"] > 0
            else 0.0
        )
        local_usage_rate = (
            (local_info["used"] / local_info["total"] * 100)
            if local_info["total"] > 0
            else 0.0
        )

        message = (
            f"📊 题库详细使用记录：\n\n"
            f"🌐 网络题库：\n"
            f"• 总数：{online_info['total']} 个谜题\n"
            f"• 已使用：{online_info['used']} 个\n"
            f"• 剩余：{online_info['available']} 个\n"
            f"• 使用率：{online_usage_rate:.1f}%\n"
            f"• 已用索引：{online_usage['used_indexes'][:10]}{'...' if len(online_usage['used_indexes']) > 10 else ''}\n\n"
            f"💾 本地存储库：\n"
            f"• 总数：{local_info['total']} 个谜题\n"
            f"• 已使用：{local_info['used']} 个\n"
            f"• 剩余：{local_info['remaining']} 个\n"
            f"• 使用率：{local_usage_rate:.1f}%\n"
            f"• 已用索引：{local_usage['used_indexes'][:10]}{'...' if len(local_usage['used_indexes']) > 10 else ''}"
        )

        yield event.plain_result(message)

    @filter.command("提示")
    async def hint_command(self, event: AstrMessageEvent):
        """根据当前所有提问记录提供方向性提示"""
        result = await self._build_hint_result(event, event.get_group_id())
        if result:
            yield result

    # 🔍 验证指令（仅在非游戏会话时处理）
    @filter.command("验证")
    async def verify_user_guess_command(self, event: AstrMessageEvent, user_guess: str):
        """验证用户推理（仅在非游戏会话时处理）"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("验证功能只能在群聊中使用")
            return

        # 检查是否有活跃游戏，如果有活跃游戏，说明在会话控制中，不在这里处理
        if self.game_state.is_game_active(group_id):
            # 阻止事件继续传播，避免被会话控制系统重复处理
            await event.block()
            return
        # 只有在没有活跃游戏时才在这里处理（用于游戏外的验证）
        yield event.plain_result("当前没有活跃的海龟汤游戏，请使用 /汤 开始新游戏")

    # ⚙️ 查看当前配置
    @filter.command("汤配置")
    async def show_config(self, event: AstrMessageEvent):
        """查看当前插件配置"""

        # 确保题库已初始化
        self._ensure_story_storages()

        local_info = self.local_story_storage.get_storage_info()
        online_info = self.online_story_storage.get_storage_info()

        # 获取策略的中文描述
        strategy_names = {
            "network_first": "优先网络题库→本地存储库→LLM生成",
            "local_first": "优先本地存储库→网络题库→LLM生成",
            "custom_first": "优先自定义题库→本地存储库→LLM生成",
            "random": "随机从网络、本地或自定义题库中选择",
        }
        strategy_name = strategy_names.get(
            self.puzzle_source_strategy, self.puzzle_source_strategy
        )

        # 检查存储库是否已满
        storage_full_warning = ""
        if local_info["available"] <= 0:
            storage_full_warning = "\n⚠️ 本地存储库已满，自动生成已停止"

        config_info = (
            f"⚙️ 海龟汤插件配置：\n"
            f"• 生成谜题 LLM：{self.generate_llm_provider_id or '默认'}\n"
            f"• 判断问答 LLM：{self.judge_llm_provider_id or '当前会话默认'}\n"
            f"• 独立复核 LLM：{self.review_llm_provider_id or '复用判断模型'}\n"
            f"• 游戏超时：{self.game_timeout} 秒\n"
            f"• 网络题库：{online_info['total']} 个谜题 (已用: {online_info['used']}, 剩余: {online_info['available']})\n"
            f"• 本地存储库：{local_info['total']}/{local_info['max_size']} (已用: {local_info['used']}, 剩余: {local_info['remaining']})\n"
            f"• 自动生成时间：{self.auto_generate_start}:00-{self.auto_generate_end}:00\n"
            f"• 谜题来源策略：{strategy_name}{storage_full_warning}"
        )
        yield event.plain_result(config_info)

    # ➕ 添加自定义海龟汤
    @filter.command("添加海龟汤")
    async def add_custom_soupai(self, event: AstrMessageEvent, content: str):
        """添加自定义海龟汤故事，格式: /添加海龟汤 <汤面>|<汤底>"""

        # 确保自定义存储库已初始化
        self._ensure_story_storages()

        # 解析内容格式: 汤面|汤底
        if "|" not in content:
            yield event.plain_result(
                "❌ 格式错误！请使用格式: /添加海龟汤 <汤面>|<汤底>"
            )
            return

        puzzle, answer = content.split("|", 1)
        puzzle = puzzle.strip()
        answer = answer.strip()

        if not puzzle or not answer:
            yield event.plain_result("❌ 汤面和汤底都不能为空！")
            return

        # 添加故事到自定义存储库
        success = self.custom_story_storage.add_story(puzzle, answer)

        if success:
            # 获取添加后的故事索引
            story_index = len(self.custom_story_storage.stories) - 1
            yield event.plain_result(
                f"✅ 添加成功！海龟汤编号: {story_index}\n\n"
                f"📖 汤面: {puzzle}\n"
                f"📖 汤底: {answer}"
            )
        else:
            yield event.plain_result("❌ 添加失败，请重试")
