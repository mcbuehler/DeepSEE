

def get_config_independent(opt):
    opt.netE = "combinedstyle"
    opt.noisy_style_scale = 0.2
    return opt


def get_config_guided(opt):
        opt.netE = "fullstyle"
        opt.noisy_style_scale = 0.05
        opt.guiding_style_image = True
        return opt


def get_opt_config(opt, name):
    if "128x128" in name and "8x_" in name:
        opt.start_size = 16
        opt.crop_size, opt.load_size = 128, 128
        opt.dataset = "celeba"
        opt.add_noise = True
    elif "256x256" in name and "8x_" in name:
        opt.start_size = 32
        opt.crop_size, opt.load_size = 256, 256
        opt.dataset = "celebamaskhq"
        opt.add_noise = True
        opt.max_fm_size = 256
    elif "32x_" in name:
        opt.start_size = 16
        opt.crop_size, opt.load_size = 512, 512
        opt.dataset = "celebamaskhq"
        opt.add_noise = False
        opt.max_fm_size = 256
    else:
        raise ValueError("Invalid name: '{}'. Please specify your options yourself.".format(name))

    if "independent" in name:
        opt = get_config_independent(opt)
    elif "guided" in name:
        opt = get_config_guided(opt)
    else:
        raise ValueError("Invalid name: '{}'. Please specify your options yourself.".format(name))
    return opt
