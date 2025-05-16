class Box:
    def __init__(self, l, w, h):
        self.l = l
        self.w = w
        self.h = h

    def orientations(self):
        return [(self.l, self.w, self.h), (self.w, self.l, self.h)]

    def key(self):
        return (self.l, self.w, self.h)


# ------------------------------
# Pallet 类
# ------------------------------
class Pallet:
    def __init__(self, l, w, h):
        self.l = l
        self.w = w
        self.h = h
        self.boxes = []
