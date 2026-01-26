import random
import time
from typing import List, Tuple, Set
from collections import defaultdict, deque
import torch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.mem_cache.cy_evict import collect_leaves, evict_core, remove_tail
import logging
import os


class RandomRadixCache(RadixCache):
    """RadixCache variant with phase marking & random token-level eviction"""
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
    ) -> None:
        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=page_size,
            disable=disable,
        )

        self.marked_count: int   = 0
        self.phase_limit: int = (
            token_to_kv_pool_allocator.size if token_to_kv_pool_allocator else 0
        )

        # for safe 
        # if self.phase_limit >= 1000:
        #     self.phase_limit -= 1000

        self.rng = random.Random()
        self.token_seen: Set[int] = set() 
        print(f"Size of kv cache is: {self.phase_limit}")

    def _on_token_access(self, node: TreeNode) -> None:
        """Book‑keep token count for phase logic."""
        if self.disable or node.id in self.token_seen:
            return
        self.token_seen.add(node.id)
        if self.marked_count >= self.phase_limit:
            self.start_new_phase(keep_node=node)

    def start_new_phase(
        self,
        keep_node: TreeNode | None = None,
    ) -> None:
        """End current phase, clear marks, roll carry_tokens to new phase."""
        if self.disable:
            return

        stack = deque([self.root_node])
        while stack:
            cur = stack.pop()
            stack.extend(cur.children.values())
            if cur is self.root_node:
                continue
            if cur.marked == True:
                cur.marked = False
                
        self.marked_count = 0

        # Reset phase counters.
        self.token_seen.clear()
        if keep_node is not None:
            self.token_seen.add(keep_node.id)
            self._mark_prefix(keep_node)    

    def _mark_node(self, node: TreeNode) -> None:
        if self.disable or node is None or node is self.root_node:
            return
        if node.marked == False:
            node.marked = True
            self.marked_count += len(node.key)
        self._on_token_access(node) 

    def _mark_prefix(self, node: TreeNode) -> None:
        if self.disable:
            return 0

        while node != self.root_node:
            if node.marked == False:
                node.marked = True
                self.marked_count += len(node.key)
                self.token_seen.add(node.id)
            node = node.parent

    def match_prefix(self, key: List[int], **kwargs) -> Tuple[torch.Tensor, int]:
        if self.disable or len(key) == 0:
            return (
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                self.root_node,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)

        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return value, last_node

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.time()

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.time()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if child.marked == False:
                    self._mark_node(new_node)
                else:
                    self.token_seen.add(new_node.id)
                    self.token_seen.add(child.id)
                    new_node.marked = True

                value.append(new_node.value)
                node = new_node
                break
            
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]
                if len(key):
                    child_key = self.get_child_key_fn(key)
        return value, node
    

    def insert(self, key: List, value=None):
        if self.disable:
            return 0
        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0
        child_key = self.get_child_key_fn(key)
        total_prefix = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.time()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)   
                if node.marked == False:
                    self._mark_node(new_node)
                else:
                    self.token_seen.add(new_node.id)
                    self.token_seen.add(node.id)
                    new_node.marked = True
                node = new_node
            else:
                self._mark_node(node)

            if len(key):
                child_key = self.get_child_key_fn(key)
                
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self._mark_node(new_node)
            self.evictable_size_ += len(value)
        return total_prefix


    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                del node.parent.children[k]
                self.evictable_size_ -= len(node.value)
                break        
    
    def evict(self, num_tokens: int) -> None:
        if self.disable or num_tokens <= 0:
            return

        if num_tokens + self.marked_count >= self.phase_limit:
            self.start_new_phase()


        leaves = []
        leaves_set = {}
        
        cap = collect_leaves(self, leaves, leaves_set)

        if len(leaves) == 0 or cap < num_tokens:
            self.start_new_phase()
            leaves = []
            leaves_set = {}
            cap = collect_leaves(self, leaves, leaves_set)

        num_evicted = 0
        tokens_to_free_list = []

        evict_core(self.rng, leaves, leaves_set, num_tokens, tokens_to_free_list, self)


        for k, v in leaves_set.items():
            if v[0] == v[1]:
                continue
            tokens_to_free_list.append(k.value[v[0]:])
            if v[0] == 0:
                self._delete_leaf(k)
            else:
                k.value = k.value[:v[0]]
                k.key = k.key[:v[0]]
                self.evictable_size_ -= (v[1] - v[0])



        if tokens_to_free_list:
            final_tokens_tensor = torch.cat(tokens_to_free_list)
            self.token_to_kv_pool_allocator.free(final_tokens_tensor)
        