def semantic_library_mode_controller(pool_addition_mode, current_gen, n_gen):
    if pool_addition_mode == "Smooth-Smallest-I":
        if current_gen > 0.5 * n_gen:
            pool_addition_mode = "Smooth-First"
        else:
            pool_addition_mode = "Smallest~Auto"
    elif pool_addition_mode == "Smooth-Smallest-I+":
        # Three stage
        if current_gen > 0.8 * n_gen:
            pool_addition_mode = "Smallest~Auto"
        elif current_gen > 0.5 * n_gen:
            pool_addition_mode = "Smooth-First"
        else:
            pool_addition_mode = "Smallest~Auto"
    elif pool_addition_mode == "Smooth-Smallest-0.8":
        if current_gen > 0.8 * n_gen:
            pool_addition_mode = "Smooth-First"
        else:
            pool_addition_mode = "Smallest~Auto"
    return pool_addition_mode
