# cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdlib cimport rand
from cpython.list cimport PyList_Append
# from sglang.srt.mem_cache.tree_node cimport TreeNode

cpdef Py_ssize_t collect_leaves(object cache,
                     list leaves_out,          
                     dict leaves_map_out):     

    cdef object node
    cdef list stack = [cache.root_node]    

    cdef Py_ssize_t l
    cdef Py_ssize_t total_capacity = 0
    while stack:
        node = stack.pop()

        if node.children:                     
            stack.extend(node.children.values())
            continue

        if (node is cache.root_node
            or node.marked == True
            or node.lock_ref > 0
            ):
            continue

        l = len(node.value)
        if l == 0:
            cache._delete_leaf(node)
            continue

        PyList_Append(leaves_out, node)
        leaves_map_out[node] = [l, l]
        total_capacity += l
    
    return total_capacity


cpdef evict_core(object rng,
                 list leaves,        
                 dict leaves_set,   
                 Py_ssize_t need,
                 list free_lst, 
                 object cache
                 ):    
    cdef Py_ssize_t size = len(leaves)
    cdef Py_ssize_t idx, remain
    while size and need:
        # idx   = rand() % size
        idx = rng.randrange(size)
        leaf  = leaves[idx]


        leaves[idx] = leaves[size-1]
        size -= 1

        if leaf not in leaves_set:
            continue

        remain = leaves_set[leaf][0]
        if remain <= 0:
            raise ValueError("picked empty leaf!")

        leaves_set[leaf][0] = remain - 1
        need -= 1

        if remain == 1:
            parent = leaf.parent                          
            free_lst.append(leaf.value)
            cache._delete_leaf(leaf)              
            del leaves_set[leaf]
            if (parent is not cache.root_node and
                parent.marked == False and
                parent.lock_ref <= 0 and
                len(parent.children) == 0):
                l = len(parent.value)
                if l > 0 and parent not in leaves_set:
                    leaves_set[parent] = [l, l]
                    leaves[size] = parent        
                    size += 1
        else:
            leaves[size] = leaf                
            size += 1

    # del leaves[size:]           
    return


cpdef remove_tail(dict leaves_set,
                  list free_lst,
                  object delete_leaf):             

    cdef object leaf
    cdef object pair
    cdef Py_ssize_t remain, orig
    cdef object view

    for leaf, pair in leaves_set.items():
        remain = <Py_ssize_t>pair[0]
        orig   = <Py_ssize_t>pair[1]
        if remain == orig:
            continue
        view = leaf.value[remain:]
        PyList_Append(free_lst, view)           
        if remain == 0:
            delete_leaf(leaf)
        else:
            leaf.value = leaf.value[:remain]      