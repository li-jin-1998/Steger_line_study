import os

from PIL import Image, ImageDraw

# 图像的尺寸
width, height = 100, 100

# 创建并保存示例图像
for i in range(4):
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 25, 75, 75), fill=(255, 0, 0), outline=(0, 0, 0))
    draw.text((40, 40), str(i + 1), fill=(0, 0, 0))
    image.save(f'image{i + 1}.png')

# 图像文件列表
image_files = ['image1.png', 'image2.png', 'image3.png', 'image4.png']

# 打开图像文件并转换为Pillow图像对象
images = [Image.open(image) for image in image_files]

# 保存为GIF文件
images[0].save(
    'output.gif',
    save_all=True,
    append_images=images[1:],
    duration=500,  # 每帧的显示时间（毫秒）
    loop=0  # 循环次数（0 表示无限循环）
)

# Remove the images
for filename in image_files:
    os.remove(filename)