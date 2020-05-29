import ffmpeg
import numpy as np
import cv2
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, 
                    default="../../GetData/youtube-8m-videos-frames-master/videos",help="输入文件夹路径")
parser.add_argument("--output", type=str, default="../datasets/Test",help="输出的数据集文件夹路径")
parser.add_argument("--Scratches",type=str,default="../../GetData/Scratches",help="划痕路径")
parser.add_argument("--p",type=float, nargs="+",default=[0.2,0.3,0.5],help="选择某种模板的概率,用'+'隔开")
parser.add_argument("--disable_FPress", default=False, action="store_true")
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--video_bitrate', type=int, default=100000)

opt = parser.parse_args()

def curve(x):
    y = 246/(1+(111/x)**3)+30
    y = np.rint(y)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return y
def SLT(im, x1, y1,x2, y2, x3, y3, x4, y4 ,x5, y5):
    lut = np.zeros(256)
    for i in range(256):
            if i < x2:
                lut[i] = ((y2-y1)/(x2-x1))*(i-x1)+y1
            elif i < x3:
                lut[i] = ((y3-y2)/(x3-x2))*(i-x2)+y2
            elif i < x4:
                lut[i] = ((y4-y3)/(x4-x3))*(i-x3)+y3
            else:
                lut[i] = ((y5-y4)/(x5-x4))*(i-x4)+y4
    img_output = cv2.LUT(im, lut)
    img_output = np.uint8(img_output+0.5)
    return img_output
def Random_SLT(im, y1):
    x1 = 0
    x2 = 90-2*y1
    y2 = 50-y1/3
    x3 = 180-11*y1/3
    y3 = x3
    x4 = 220-2*y1
    y4 = 210-y1/3
    x5 = 255
    y5 = 225+y1
    lut = np.zeros(256)
    for i in range(256):
            if i < x2:
                lut[i] = ((y2-y1)/(x2-x1))*(i-x1)+y1
            elif i < x3:
                lut[i] = ((y3-y2)/(x3-x2))*(i-x2)+y2
            elif i < x4:
                lut[i] = ((y4-y3)/(x4-x3))*(i-x3)+y3
            else:
                lut[i] = ((y5-y4)/(x5-x4))*(i-x4)+y4
    img_output = cv2.LUT(im, lut)
    img_output = np.uint8(img_output+0.5)
    return img_output
def video_multiply(x, y):
    x = np.int32(x)
    y = np.int32(y)
    out = x*y/255
    out = np.uint8(out)
    return out
def png_add(v_im, m_im):
    b,g,r,a = cv2.split(m_im)
    fg = cv2.merge((b,g,r))
    mask = cv2.merge((a,a,a))
    mask_inv = cv2.bitwise_not(a)
    bg = cv2.bitwise_and(v_im, v_im, mask=mask_inv)
    fg = cv2.bitwise_and(fg, fg, mask=a)
    return cv2.add(fg, bg)
def gaussian_noise(im, mean=0, var=0.001):
    im = np.array(im/255, dtype=np.float)
    noise = np.random.normal(mean, var**5, im.shape)
    out = im+noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    return out

## ffmpeg with mask
# def FPress(input, mask, output='pipe:', format='rawvideo'):
#     probe = ffmpeg.probe(input)
#     video_info = next(s for s in probe['streams'] if s['codec_type']=='video')
#     width = int(video_info['width'])
#     height = int(video_info['height'])
#     num_frames = int(video_info['nb_frames'])
#     v_in = ffmpeg.input(input)
#     v_mask = (
#         ffmpeg.input(mask)
#         .filter('scale', width,height)
#         # .colorchannelmixer(aa=0)
#     )
#     # v_alpha = v_in.filter('alphamerge', v_mask)
#     out, err = (
#         v_in
#         .overlay(v_mask) 
#         # .filter('lutyuv', u=128, v=128)
#         .filter('fps', fps=25)
#         .output(output,video_bitrate=200000, format=format, pix_fmt='gray',  t=20)
#         .overwrite_output()
#         .run(capture_stdout=True)
#     )
#     frames = np.frombuffer(out, dtype=np.uint8).reshape((-1, height,width))
#     return frames, height, width
def FPress(input, output, video_name):
    probe = ffmpeg.probe(input)
    video_info = next(s for s in probe['streams'] if s['codec_type']=='video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    (
        ffmpeg.input(input)
        .filter('fps', fps=25)
        .output(output+"/HR/"+video_name+"/%06d.png", pix_fmt='gray',ss=20, t=10)
        .overwrite_output()
        .run(capture_stdout=True)
    )
    (
        ffmpeg.input(input)
        .filter('fps', fps=25)
        .filter("scale",width/opt.upscale_factor,height/opt.upscale_factor)
        .output(output+"/tmp/output.mp4", video_bitrate=100000, pix_fmt='gray',ss=20,t=10)
        .overwrite_output()
        .run(capture_stdout=True)
    )
    # frames = np.frombuffer(out, dtype=np.uint8).reshape((-1, height,width))

def prepare_mask(path):
    mask_types = ["png","black","white"]
    mask_type = np.random.choice(mask_types,p=opt.p)
    path = path+'/'+mask_type
    mask_list = os.listdir(path)
    mask_name = random.choice(mask_list)
    path = os.path.join(path,mask_name)
    print("Applying:"+path)
    if mask_type=="png":
        mlist = os.listdir(path)
        while mlist:
            m_name = os.path.join(path,mlist.pop(0))
            m_im = cv2.imdecode(np.fromfile(m_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            yield mask_type, m_im
    else:
        mask = cv2.VideoCapture(path)
        m_ret, m_im = mask.read(cv2.IMREAD_UNCHANGED)
        yield mask_type, m_im
        while m_ret:
            m_ret, m_im = mask.read(cv2.IMREAD_UNCHANGED)
            yield mask_type, m_im

def process_single_video(input, output, opt=opt):  
   
    if not os.path.exists(output+"/tmp"):
        os.mkdir(output+"/tmp")
    video_name = os.path.basename(input).split('.')[0]
    if not os.path.exists(output+"/HR/"+video_name):
        os.makedirs(output+"/HR/"+video_name, exist_ok=True)
    if not os.path.exists(output+"/LR/"+video_name):
        os.makedirs(output+"/LR/"+video_name, exist_ok=True)
    
    if not opt.disable_FPress:
        FPress(input, output, video_name)

    #本地读取文件
    cap = cv2.VideoCapture(output+"/tmp/output.mp4")
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fcount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    v_ret, v_im = cap.read()
    index = 1
    ## seeds
    slt_seed = random.uniform(0,30.0)
    noise_seed = random.uniform(0,0.5)
    blur_seed = random.uniform(0.6,1.2)
    print("slt_seed(y1):{:.4f}\nnoise_seed:{:.4f}\nblur_seed:{:.4f}".format(slt_seed, noise_seed, blur_seed))
    mask_generator = prepare_mask(opt.Scratches)
    while v_ret:
        ## 应用素材
        try:
            mask_type, m_im = next(mask_generator)
            if not m_im.size:
                mask_type, m_im = next(mask_generator)
        except:
            print("Changing Mask...")
            mask_generator = prepare_mask(opt.Scratches)
            mask_type, m_im = next(mask_generator)
        m_im = cv2.resize(m_im, (v_width, v_height))
        if mask_type=="png":
            v_im = png_add(v_im, m_im)
            v_im = cv2.cvtColor(v_im, cv2.COLOR_BGRA2GRAY)
        else:
            v_im = cv2.cvtColor(v_im, cv2.COLOR_BGR2GRAY)
            m_im = cv2.cvtColor(m_im, cv2.COLOR_BGR2GRAY)
            if mask_type=="black":
                v_im = cv2.add(v_im, m_im)
            else:
                v_im = video_multiply(v_im, m_im)

        #灰度转换
        # v_im = curve(v_im)
        v_im = Random_SLT(v_im, slt_seed)
        
        #高斯噪声
        v_im = gaussian_noise(v_im, var=noise_seed)

        #高斯模糊
        v_im = cv2.GaussianBlur(v_im, (0,0), blur_seed)
        v_im = cv2.cvtColor(v_im,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(output+"/LR/"+video_name+"/{:06d}.png".format(index), v_im)
        print("frame:{}/{}".format(index, fcount), end='\r')
        index +=1
        v_ret, v_im = cap.read()
    cap.release()
    cv2.destroyAllWindows()

for i in os.listdir(opt.input):
    if i.endswith(".mp4"):
        print("===>Processing "+i)
        process_single_video(os.path.join(opt.input,i),opt.output,opt)
