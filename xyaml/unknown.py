# Ignore tags with no defined constructor and handle recursion

import yaml
import wrapt


UNSET = object()


# https://death.andgravity.com/any-yaml
class Tagged(wrapt.ObjectProxy):
    tag: str = None

    def __init__(self, tag: str, value: object = UNSET):
        if value is not UNSET:
            super().__init__(value)
        self.tag = tag

    def set_value(self, value):
        super().__init__(value)

    def __repr__(self):
        return f"{type(self).__name__}({self.tag!r}, {self.__wrapped__!r})"


def construct_undefined(self: yaml.Loader, node: yaml.Node):

    if isinstance(node, yaml.nodes.ScalarNode):
        yield Tagged(node.tag, self.construct_scalar(node))
    else:
        child = Tagged(node.tag)
        yield child  # https://stackoverflow.com/questions/27826576/how-do-i-handle-recursion-in-a-custom-pyyaml-constructor

        if isinstance(node, yaml.nodes.SequenceNode):
            child.set_value(self.construct_sequence(node))
        elif isinstance(node, yaml.nodes.MappingNode):
            child.set_value(self.construct_mapping(node))


def register_undefined(loader: yaml.Loader = yaml.SafeLoader):
    yaml.add_constructor(None, construct_undefined, loader)
