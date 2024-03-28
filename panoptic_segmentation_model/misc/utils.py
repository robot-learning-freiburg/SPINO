from yacs.config import CfgNode

_VALID_TYPES = {tuple, list, str, int, float, bool, None}

def convert_to_dict(cfg_node, key_list=None):
    """ Convert a config node to dictionary """

    key_list = [] if key_list is None else key_list

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(f"Key {'.'.join(key_list)} with value {type(cfg_node)} is not a valid type; "
                  f"valid types: {_VALID_TYPES}")
        return cfg_node

    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
        cfg_dict[k] = convert_to_dict(v, key_list + [k])
    return cfg_dict
