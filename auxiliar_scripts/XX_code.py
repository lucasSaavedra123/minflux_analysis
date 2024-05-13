import moviepy.editor as mp
from moviepy.video.fx.all import crop
from moviepy.editor import *

clip = mp.VideoFileClip("animation.gif")
(w, h) = clip.size
clip = crop(clip, width=h, height=h, x_center=w/2, y_center=h/2)
clip = clip.resize(0.5)
clip = clip.subclip(0, 5)


clips = [[clip, clip.rotate(90), clip.rotate(180)]]
clip = clips_array(clips)
clip.write_videofile('animation.mp4')
