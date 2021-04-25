"""
Based on https://github.com/clemtoy/pptree/blob/master/pptree/pptree.py   (That one has a MIT License)
Creates a nice string based on a tree structure
The pretty-print function

tree2string(current_node, childattr='children', nameattr='name')

    the root node object
    the name of the list containing the children (optional)
    the name of the field containing the text to display. If nameattr is not filled and the custom node don't have any name field, then the str function is used. (optional)

Example using provided Node class

from pptree import *

shame = Node("shame")

conscience = Node("conscience", shame)
selfdisgust = Node("selfdisgust", shame)
embarrassment = Node("embarrassment", shame)

selfconsciousness = Node("selfconsciousness", embarrassment)
shamefacedness = Node("shamefacedness", embarrassment)
chagrin = Node("chagrin", embarrassment)
discomfiture = Node("discomfiture", embarrassment)
abashment = Node("abashment", embarrassment)
confusion = Node("confusion", embarrassment)

print_tree(shame)
"""


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

        if parent:
            self.parent.children.append(self)


def tree2string(current_node, childattr='children', nameattr='name', indent='', last='updown'):
    def tree2linelist(current_node, childattr, nameattr, indent, last, lines):

        if hasattr(current_node, nameattr):
            name = lambda node: getattr(node, nameattr)
        else:
            name = lambda node: str(node)

        children = lambda node: getattr(node, childattr)
        nb_children = lambda node: sum(nb_children(child) for child in children(node)) + 1
        size_branch = {child: nb_children(child) for child in children(current_node)}

        """ Creation of balanced lists for "up" branch and "down" branch. """
        up = sorted(children(current_node), key=lambda node: nb_children(node))
        down = []
        while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
            down.append(up.pop())

        """ Printing of "up" branch. """
        for child in up:
            next_last = 'up' if up.index(child) is 0 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', ' ' * len(f"{name(current_node)}"))
            tree2linelist(child, childattr, nameattr, next_indent, next_last, lines)

        """ Printing of current node. """
        if last == 'up':
            start_shape = '┌'
        elif last == 'down':
            start_shape = '└'
        elif last == 'updown':
            start_shape = ' '
        else:
            start_shape = '├'

        if up:
            end_shape = '┤'
        elif down:
            end_shape = '┐'
        else:
            end_shape = ''

        lines.append('{0}{1}{2}{3}'.format(indent, start_shape, name(current_node), end_shape))

        """ Printing of "down" branch. """
        for child in down:
            next_last = 'down' if down.index(child) is len(down) - 1 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else '│', ' ' * len(f"{name(current_node)}"))
            tree2linelist(child, childattr, nameattr, next_indent, next_last, lines)

    lines = []
    tree2linelist(current_node=current_node, childattr=childattr, nameattr=nameattr, indent=indent, last=last,
                  lines=lines)

    return "\n".join(lines)


if __name__ == "__main__":
    shame = Node("shame")

    conscience = Node("conscience", shame)
    selfdisgust = Node("selfdisgust", shame)
    embarrassment = Node("embarrassment", shame)

    selfconsciousness = Node("selfconsciousness", embarrassment)
    shamefacedness = Node("shamefacedness", embarrassment)
    chagrin = Node("chagrin", embarrassment)
    discomfiture = Node("discomfiture", embarrassment)
    abashment = Node("abashment", embarrassment)
    confusion = Node("confusion", embarrassment)

    str = tree2string(shame)
    print(str)
