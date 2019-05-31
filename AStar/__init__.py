from .bin import module_path as __origin__  # Add proper CorePyExt's path to sys path
from Module import Direction, Board, AStar

del bin  # Clear the intermediary module
__doc__ = f"C++ extension 'AStar' with origin path at '{__origin__}'"
