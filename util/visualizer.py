class Visualizer(object):
    def __init__(self, opt):
        if opt.display_id > 0:
            import visdom
            self.vis = visdom.Visdom()

    def display_images(self, visuals, epoch):
        idx = 1
        for label, image_numpy in visuals.iteritems():
            self.vis.image()
