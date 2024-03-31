from evolutionary_forest.component.configuration import Configuration


def consistency_check(learner):
    global_dict = {}
    for attribute_name, attribute in learner.__dict__.items():
        global_dict[attribute_name] = attribute
        if isinstance(attribute, Configuration):
            if hasattr(
                attribute, "__dict__"
            ):  # Check if attribute is an object with attributes
                for key, value in attribute.__dict__.items():
                    if key not in global_dict:
                        global_dict[key] = value
                    else:
                        if value != global_dict[key]:
                            raise ValueError(
                                f"Configuration attribute {key} is not consistent"
                            )
