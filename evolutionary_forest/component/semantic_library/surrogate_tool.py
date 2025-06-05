import copy


def remap_gp_tree_variables(expr, mapping, pset):
    # Prepare variable map: 'x0' -> terminal object
    var_map = {
        t.value: t
        for t in pset.terminals[object]
        if hasattr(t, "value") and isinstance(t.value, str) and t.value.startswith("x")
    }
    expr_new = copy.deepcopy(expr)
    for i, node in enumerate(expr_new):
        if (
            hasattr(node, "value")
            and isinstance(node.value, str)
            and node.value.startswith("x")
        ):
            old_idx = int(node.value[1:])
            if old_idx < len(mapping):
                new_var = f"x{mapping[old_idx]}"
                if new_var in var_map:
                    expr_new[i] = var_map[new_var]
    return expr_new
