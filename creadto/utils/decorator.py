import ast
import inspect
import functools
import textwrap


_singleton_instances = {}

def service(cls):
    def check_no_self_assignment(func):
        try:
            source_code = inspect.getsource(func)
            source_code = textwrap.dedent(source_code)
        except OSError:
            return

        tree = ast.parse(source_code)

        class SelfAssignmentChecker(ast.NodeVisitor):
            def __init__(self):
                self.found = []

            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        self.found.append(target.attr)

        checker = SelfAssignmentChecker()
        checker.visit(tree)

        if checker.found:
            raise SyntaxError(f"Class '{cls.__name__}' modifies self inside __call__: {checker.found}")

    if hasattr(cls, "__call__"):
        check_no_self_assignment(cls.__call__)

    def get_instance(*args, **kwargs):
        if cls not in _singleton_instances:
            _singleton_instances[cls] = cls(*args, **kwargs)
        return _singleton_instances[cls]

    return get_instance

def get_instances(cls):
    return _singleton_instances.get(cls, None)
