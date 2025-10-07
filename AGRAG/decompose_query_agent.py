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

class SimpleQueryDecompositionAgent:
    
    def __init__(self, max_iterations: int = 2):
        self.name = "QueryDecomposer"
        self.max_iterations = max_iterations
        self._numbered_pattern = re.compile(r'^\d+\.\s*(.+)')
        self._bullet_pattern = re.compile(r'^-\s*(.+)')
    
    async def execute(self, query: str, max_sub: int = 4) -> SimpleAgentState:
        state = SimpleAgentState(
            query=query,
            context={"max_sub": max_sub},
            history=[],
            confidence=0.0,
            iterations=0,
            max_iterations=self.max_iterations
        )
        
        logger.info(f"[{self.name}] Starting decomposition for: {query}")
        
        best_subqueries = []
        best_score = 0.0
        
        while state.iterations < self.max_iterations:
            state.iterations += 1
            
            strategy = await self._detect_query_category(query)
                  
            logger.info(f"[{self.name}] Iteration {state.iterations}: Using strategy '{strategy}'")
            
            try:
                subqueries = await asyncio.wait_for(
                    self._decompose_with_strategy(state, strategy), 
                    timeout=25.0
                )
                if not subqueries:
                    logger.warning(f"[{self.name}] No subqueries generated, returning original query")
                    return self._finalize_state(state, [query])
                    
            except asyncio.TimeoutError:
                logger.warning(f"[{self.name}] Timeout, returning original query")
                return self._finalize_state(state, [query])
            
            is_good, reasoning, score = self._evaluate_decomposition_fast(state, subqueries)
            state.history.append(f"Iteration {state.iterations}: {strategy} -> {len(subqueries)} subqueries (score: {score:.2f})")
            
            if score > best_score:
                best_subqueries = subqueries
                best_score = score
            
            if is_good and score > 0.6:
                state.confidence = score
                logger.info(f"[{self.name}] Early exit (confidence: {state.confidence:.3f})")
                return self._finalize_state(state, subqueries)
            
            if state.iterations == 1 and score > 0.5:
                logger.info(f"[{self.name}] First attempt acceptable, skipping retry")
                break
        
        state.confidence = best_score if best_subqueries else 0.5
        final_result = best_subqueries if best_subqueries else [query]
        logger.info(f"[{self.name}] Returning {'decomposed queries' if best_subqueries else 'original query'} (confidence: {state.confidence:.3f})")
        return self._finalize_state(state, final_result)
    
    async def _detect_query_category(self, query: str) -> str:
        prompt = f"""Classify this query into ONE category:

        Query: {query}
        
        Categories:
        - categorical: Multi-aspect questions needing different perspectives  
        - hierarchical: Complex topics needing breakdown from general to specific
        - sequential: Process/procedure questions needing step-by-step answers
        - simple: Basic factual questions (what, who, when, where)
        
        Respond with only the category name:"""
        
        try:
            response = await gpt_o4_mini_complete(prompt)
            # response = await gemini_reason(prompt)
            category = response.strip().lower()
            logger.info(f"[{category}] found")
            if category in ['categorical', 'hierarchical', 'sequential', 'simple']:
                return category
            else:
                raise ValueError(f"Invalid category: {category}")
        except Exception as e:
            raise   
    

    
    async def _decompose_with_strategy(self, state: SimpleAgentState, strategy: str) -> List[str]:
        max_sub = state.context['max_sub']
        query = state.query
        
        strategy_prompts = {
            
            'categorical': f"""Break this query into exactly {max_sub} different aspect categories:
            
            Query: {query}
            
            Focus on different perspectives, types, categories, or dimensions of the topic.
            Generate exactly {max_sub} questions covering different aspects.
            
            Return as JSON array: ["question 1", "question 2", ...]""",
            
            'hierarchical': f"""Break this query into exactly {max_sub} questions from general to specific:
            
            Query: {query}
            
            Start with broad concepts, then drill down to specific details.
            Generate exactly {max_sub} questions in hierarchical order.
            
            Return as JSON array: ["question 1", "question 2", ...]""",

            'sequential': f"""Break this query into exactly {max_sub} sequential logical steps:

            Query: {query}
            
            Focus on step-by-step process or logical flow to answer the query.
            Generate exactly {max_sub} questions in sequential order.
            
            Return as JSON array: ["question 1", "question 2", ...]""",

            'simple': f"""Break this query into exactly {max_sub} direct, factual sub-questions:

            Query: {query}
            
            Focus on basic factual elements: what, who, when, where, why, how.
            Generate exactly {max_sub} clear, simple questions.
            
            Return as JSON array: ["question 1", "question 2", ...]"""
        }
        
        prompt = strategy_prompts.get(strategy)
        raw_output = await gpt_o4_mini_complete(prompt)
        # raw_output = await gemini_reason(prompt)
        return self._parse_subqueries_fast(raw_output, max_sub)
    
    def _parse_subqueries_fast(self, raw_output: str, target_count: int) -> List[str]:
        try:
            cleaned = raw_output.strip().replace('```json', '').replace('```', '')
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                subqueries = [str(q).strip() for q in parsed if str(q).strip()]
                return subqueries[:target_count] if len(subqueries) >= target_count else subqueries
        except json.JSONDecodeError:
            pass
        
        subqueries = []
        for line in raw_output.splitlines():
            line = line.strip()
            
            match = self._numbered_pattern.match(line)
            if match:
                subqueries.append(match.group(1).strip())
                continue
                
            match = self._bullet_pattern.match(line)
            if match:
                subqueries.append(match.group(1).strip())
                continue
                
            if line.endswith('?') and len(line.split()) > 3:
                subqueries.append(line)
        
        seen = set()
        unique = []
        for q in subqueries:
            if q not in seen and len(unique) < target_count:
                seen.add(q)
                unique.append(q)
        
        return unique
    
    def _evaluate_decomposition_fast(self, state: SimpleAgentState, subqueries: List[str]) -> Tuple[bool, str, float]:
        target_count = state.context.get('max_sub', 4)
        actual_count = len(subqueries)
        
        if actual_count == 0:
            return False, "No subqueries generated", 0.1
        
        if actual_count != target_count:
            return False, f"Generated {actual_count} but need {target_count}", 0.4
        
        score = 0.5 
    
        avg_length = sum(len(q.split()) for q in subqueries) / actual_count
        if 4 <= avg_length <= 12:
            score += 0.2
        
        question_count = sum(1 for q in subqueries if '?' in q)
        if question_count >= actual_count * 0.5:
            score += 0.15
        
        unique_starts = len(set(q[:10].lower() for q in subqueries))
        if unique_starts == actual_count:
            score += 0.15
        
        is_adequate = score > 0.7
        reasoning = f"Fast check: count={actual_count}/{target_count}, avg_words={avg_length:.1f}, questions={question_count}"
        
        return is_adequate, reasoning, score
    
    def _finalize_state(self, state: SimpleAgentState, subqueries: List[str]) -> SimpleAgentState:
        state.context["result"] = subqueries
        state.context["final_count"] = len(subqueries)
        return state

_cached_decomposer = None

async def decompose_query(query: str, max_sub: int = 4) -> List[str]:
    global _cached_decomposer
    if _cached_decomposer is None:
        _cached_decomposer = SimpleQueryDecompositionAgent()
    
    state = await _cached_decomposer.execute(query, max_sub)
    
    logger.info(f"Query decomposition completed: {state.context['final_count']} subqueries generated "
                f"(confidence: {state.confidence:.3f})")
    
    return state.context.get("result", [query])