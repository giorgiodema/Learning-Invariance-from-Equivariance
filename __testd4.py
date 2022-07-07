from groupconv.groups import *
from PIL import Image

d4 = DihedralGroup(4)
elems = d4.elements()
inv = d4.inverse(elems)
new_elems = elems.clone()
new_elems = d4.left_action_on_H(elems,new_elems)
print(new_elems)