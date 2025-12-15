"""Block-based compression primitives: uniform grid, quadtree, and k-d tree."""

import numpy as np


def compress_uniform_grid(analyzer, image_size, block_size):
    # Simplest mode: fixed-size tiles colored by their mean.
    if block_size == 0: return [], 0
    width, height = image_size
    output_blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Fixed-size tiling; no variance check, so detail loss is uniform.
            box = (x, y, min(x + block_size, width), min(y + block_size, height))
            avg_color, _ = analyzer.get_region_stats(box)
            output_blocks.append((box, avg_color))
    return output_blocks, len(output_blocks)


class QuadtreeNode:
    def __init__(self, box):
        self.box, self.children, self.is_leaf, self.color = box, None, False, None


def build_quadtree(analyzer, box, threshold, depth, max_depth):
    node = QuadtreeNode(box)
    avg_color, variance = analyzer.get_region_stats(box)
    node.color = avg_color
    # Stop if region is flat enough, tiny, or depth limit reached.
    if variance < threshold or (box[2] - box[0]) <= 8 or depth >= max_depth:
        node.is_leaf = True
        return node
    x1, y1, x2, y2 = box
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    boxes = [(x1, y1, mid_x, mid_y), (mid_x, y1, x2, mid_y), (x1, mid_y, mid_x, y2), (mid_x, mid_y, x2, y2)]
    node.children = [build_quadtree(analyzer, b, threshold, depth + 1, max_depth) for b in boxes if b[0] < b[2] and b[1] < b[3]]
    return node


def get_quadtree_leaves(node):
    if node is None: return []
    if node.is_leaf: return [node]
    leaves = []
    if node.children:
        for child in node.children: leaves.extend(get_quadtree_leaves(child))
    return leaves


def compress_quadtree(analyzer, image_size, threshold):
    # Depth cap prevents over-fragmentation on large images.
    max_depth = min(12, int(np.log2(min(image_size))))
    root = build_quadtree(analyzer, (0, 0, image_size[0], image_size[1]), threshold, 0, max_depth)
    leaves = get_quadtree_leaves(root)
    return [(leaf.box, leaf.color) for leaf in leaves], len(leaves)


class KDTreeNode:
    def __init__(self, box):
        self.box, self.left, self.right, self.is_leaf, self.color = box, None, None, False, None


def build_kdtree(analyzer, box, threshold, depth, max_depth):
    node = KDTreeNode(box)
    avg_color, variance = analyzer.get_region_stats(box)
    node.color = avg_color
    # Stop when region is flat, thin, or deep enough.
    if variance < threshold or (box[2] - box[0]) <= 8 or (box[3] - box[1]) <= 8 or depth >= max_depth:
        node.is_leaf = True
        return node
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    # Split along the longer axis to balance aspect ratios.
    if width > height:
        split = (x1 + x2) // 2
        box1, box2 = (x1, y1, split, y2), (split, y1, x2, y2)
    else:
        split = (y1 + y2) // 2
        box1, box2 = (x1, y1, x2, split), (x1, split, x2, y2)
    if box1[0] < box1[2] and box1[1] < box1[3]:
        node.left = build_kdtree(analyzer, box1, threshold, depth + 1, max_depth)
    if box2[0] < box2[2] and box2[1] < box2[3]:
        node.right = build_kdtree(analyzer, box2, threshold, depth + 1, max_depth)
    return node


def get_kdtree_leaves(node):
    if node is None: return []
    if node.is_leaf: return [node]
    leaves = []
    leaves.extend(get_kdtree_leaves(node.left))
    leaves.extend(get_kdtree_leaves(node.right))
    return leaves


def compress_kdtree(analyzer, image_size, threshold):
    # Slightly deeper cap than quadtree to allow finer directional splits.
    max_depth = min(14, int(np.log2(min(image_size))) + 4)
    root = build_kdtree(analyzer, (0, 0, image_size[0], image_size[1]), threshold, 0, max_depth)
    leaves = get_kdtree_leaves(root)
    return [(leaf.box, leaf.color) for leaf in leaves], len(leaves)
