import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
from .llm import gemini_reason, gpt_o4_mini_complete
from .utils import logger

@dataclass
class SimpleAgentState:
    query: str
    context: Dict[str, Any]
    history: List[str]
    confidence: float
    iterations: int
    max_iterations: int = 3

class SimplePathPruningAgent:
    
    def __init__(self, max_iterations: int = 2):
        self.name = "PathPruner"
        self.max_iterations = max_iterations
        self._path_pattern = re.compile(r'"[^"]+"(?:\s*->\s*"[^"]+")+')
    
    async def execute(self, query: str, paths: List[List[str]]) -> SimpleAgentState:
        if not paths:
            return SimpleAgentState(
                query=query,
                context={"original_paths": [], "path_count": 0, "result": [], "final_count": 0, "retention_rate": 0},
                history=["No paths to prune"],
                confidence=1.0,
                iterations=0,
                max_iterations=self.max_iterations
            )
        
        if len(paths) <= 3:
            return SimpleAgentState(
                query=query,
                context={"original_paths": paths, "path_count": len(paths), "result": paths, "final_count": len(paths), "retention_rate": 1.0},
                history=["Small path set - no pruning needed"],
                confidence=0.9,
                iterations=0,
                max_iterations=self.max_iterations
            )
        
        state = SimpleAgentState(
            query=query,
            context={"original_paths": paths, "path_count": len(paths)},
            history=[],
            confidence=0.0,
            iterations=0,
            max_iterations=self.max_iterations
        )
        
        logger.info(f"[{self.name}] Starting path pruning for {len(paths)} paths")
        
        best_pruned_paths = []
        best_score = 0.0
        
        while state.iterations < self.max_iterations:
            state.iterations += 1
            
            try:
                pruned_paths = await asyncio.wait_for(self._prune_paths(state), timeout=25.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.name}] Timeout, using fallback")
                pruned_paths = self._quick_fallback_prune(paths, query)
          
            is_good, reasoning, score = self._evaluate_pruning_fast(state, pruned_paths)
            state.history.append(f"Iteration {state.iterations}: {len(pruned_paths)}/{len(paths)} paths kept (score: {score:.2f})")
            
            if score > best_score:
                best_pruned_paths = pruned_paths
                best_score = score
            
           
            if is_good and score > 0.65:
                state.confidence = score
                logger.info(f"[{self.name}] Early exit (confidence: {state.confidence:.3f})")
                return self._finalize_state(state, pruned_paths)
            
            if state.iterations == 1 and score > 0.5:
                logger.info(f"[{self.name}] First attempt acceptable, skipping retry")
                break
        
        state.confidence = best_score
        logger.info(f"[{self.name}] Returning best attempt (confidence: {state.confidence:.3f})")
        return self._finalize_state(state, best_pruned_paths if best_pruned_paths else paths)
    
    def _quick_fallback_prune(self, paths: List[List[str]], query: str) -> List[List[str]]:
        query_words = set(query.lower().split())
        scored_paths = []
        
        for path in paths:
            path_text = " ".join(path).lower()
            score = sum(1 for word in query_words if word in path_text)
            scored_paths.append((score, path))
        
        scored_paths.sort(reverse=True, key=lambda x: x[0])
        keep_count = max(1, min(len(paths) // 2, len(paths) - 1))
        return [path for _, path in scored_paths[:keep_count]]
    
    async def _prune_paths(self, state: SimpleAgentState) -> List[List[str]]:
        paths = state.context["original_paths"]
        candidates = [" -> ".join(path) for path in paths]
        original_count = len(paths)
        
        prompt = f"""Filter these graph paths for query relevance. Keep 30-70% most relevant paths.

        Query: {state.query}

        Paths:
        {chr(10).join(f"{i+1}. {p}" for i, p in enumerate(candidates))}

        Return *only* the relevant paths as a **JSON array of strings**, each string being a path in the format "A -> B -> C".
        Do **not** include any markdown, explanation, or additional text — only the JSON.
        """

        raw_response = await gpt_o4_mini_complete(prompt)
        # raw_response = await gemini_reason(prompt)
        
        pruned_paths = self._parse_response_fast(raw_response)
        
        if not pruned_paths or len(pruned_paths) > original_count:
            logger.warning(f"[{self.name}] Parse failed, using heuristic")
            return self._quick_fallback_prune(paths, state.query)
        
        return pruned_paths
    
    def _parse_response_fast(self, raw_response: str) -> List[List[str]]:
        cleaned = raw_response.strip().replace('```json', '').replace('```', '')
        
        try:
            arr = json.loads(cleaned)
            if isinstance(arr, list):
                return self._convert_to_paths(arr)
        except json.JSONDecodeError:
            pass
        
        matches = self._path_pattern.findall(cleaned)
        if matches:
            return self._convert_to_paths(matches)
        
        lines = [line.strip(' -"') for line in cleaned.split('\n') if '->' in line]
        return self._convert_to_paths(lines) if lines else []
    
    def _convert_to_paths(self, path_strings: List[str]) -> List[List[str]]:
        result = []
        for path_str in path_strings:
            if isinstance(path_str, str) and '->' in path_str:
                parts = [seg.strip().strip('"\'') for seg in path_str.split('->')]
                quoted = [f'"{part}"' for part in parts if part]
                if quoted:
                    result.append(quoted)
        return result
    
    def _evaluate_pruning_fast(self, state: SimpleAgentState, pruned_paths: List[List[str]]) -> Tuple[bool, str, float]:
        original_count = len(state.context["original_paths"])
        pruned_count = len(pruned_paths)
        
        if pruned_count == 0:
            return False, "No paths kept", 0.1
        
        retention_rate = pruned_count / original_count
        
        if 0.25 <= retention_rate <= 0.75:
            return True, f"Good retention: {retention_rate:.1%}", 0.8
        elif 0.1 <= retention_rate < 0.25:
            return True, f"Aggressive: {retention_rate:.1%}", 0.7
        elif retention_rate < 0.1:
            return False, f"Too aggressive: {retention_rate:.1%}", 0.3
        else:
            return retention_rate <= 0.95, f"Conservative: {retention_rate:.1%}", 0.6
    
    def _finalize_state(self, state: SimpleAgentState, pruned_paths: List[List[str]]) -> SimpleAgentState:
        state.context["result"] = pruned_paths
        state.context["final_count"] = len(pruned_paths)
        state.context["retention_rate"] = len(pruned_paths) / len(state.context["original_paths"]) if state.context["original_paths"] else 0
        return state


_cached_agent = None

async def prune_irrelevant_paths(query: str, paths: List[List[str]]) -> List[List[str]]:
    global _cached_agent
    if _cached_agent is None:
        _cached_agent = SimplePathPruningAgent()
    
    state = await _cached_agent.execute(query, paths)
    
    logger.info(f"Path pruning completed: {state.context['final_count']}/{state.context['path_count']} paths kept "
                f"(retention: {state.context['retention_rate']:.1%}, confidence: {state.confidence:.3f})")
    
    return state.context.get("result", paths)