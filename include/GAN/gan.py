def predict():
    import os
    path = os.getcwd() + '/src/navigation_explainer/include/GAN'
    import sys
    sys.path.insert(1, path)


    from options.test_options import TestOptions
    from models import create_model
    from util.util import tensor2im

    import PIL.Image
    import matplotlib.pyplot as plt
    import numpy as np

    #rgb_image = PIL.Image.open('/home/amar/amar_ws/input.png')
    #rgb_image = rgba_image.convert('RGB')
    #plt.imshow(rgb_image)

    path = os.getcwd() + '/input.png'
    input = PIL.Image.open(path).convert('RGB')
    #plt.imshow(input)
    #plt.savefig('input_main.png')


    #'''            
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #print(type(opt))
    #'''

    #'''
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    #'''

    #input_np = np.array(input)
    #print(input_np.shape)
    #plt.imshow(input_np)
    #plt.savefig('input_main_np.png')


    from data.base_dataset import get_params, get_transform
    transform_params = get_params(opt, input.size)
    input_nc = 3
    input_transform = get_transform(opt, transform_params, grayscale=(input_nc == 1))       
    input = input_transform(input)
    #print(type(input))

    model.set_input_one(input)  # unpack data from data loader
    model.forward()

    output = tensor2im(model.fake_B)
    fig = plt.figure(frameon=False)
    w = 1.6 #* 3
    h = 1.6 #* 3
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(output, aspect='auto')
    fig.savefig('GAN.png')
    fig.clf()
