import webdataset as wds
import os
import cv2
import sys
from tqdm import trange, tqdm
import tarfile

def get_last_image_output_path(tar_path, output_dir):
    """
    从 tar 包中读取最后一个图片成员的输出路径（jpg）
    """
    valid_exts = {".jpg"}
    with tarfile.open(tar_path, 'r') as tar:
        # 只取普通文件且是图片
        members = [m for m in tar.getmembers() 
                   if m.isfile() and os.path.splitext(m.name)[1].lower() in valid_exts]
        if not members:
            return None
        last_member = members[-1]

        # WebDataset的__key__不带扩展名，这里去掉文件扩展名来模拟key
        key = os.path.splitext(last_member.name)[0]
        base_dir = os.path.dirname(key).replace("insta-train-vitpose-replicate", "insta-train")
        filename = os.path.basename(key) + ".jpg"  # 强制输出jpg
        image_path_dir = os.path.join(output_dir, base_dir)
        return os.path.join(image_path_dir, filename)
    
def extract_images(start_index, end_index, tar_url_template, output_dir):
    start = int(start_index)
    end = int(end_index)

    for idx in trange(start, end + 1):
        tar_path = tar_url_template.format(i=idx)
        if not os.path.exists(tar_path):
            print(f"Tar file not found: {tar_path}, skipping...")
            continue

        # --------------- 检查最后一个文件 ---------------
        last_file_path = get_last_image_output_path(tar_path, output_dir)
        if last_file_path is None:
            print(f"No valid files in {tar_path}, skipping...")
            continue
        
        print(last_file_path)

        if os.path.exists(last_file_path):
            print(f"Tar {tar_path} seems already fully extracted (last file exists), skipping...")
            continue

        # Start extracting
        print(f"Extracting tar {tar_path}...")
        dataset = wds.WebDataset(tar_path).decode("rgb8").rename(jpg="jpg;jpeg;png")

        for i, sample in tqdm(enumerate(dataset), desc=f"Extracting {idx:06d}"):
            try:
                key = sample["__key__"]
                image_data = sample["jpg"]

                base_dir = os.path.dirname(key).replace("insta-train-vitpose-replicate", "insta-train")
                filename = os.path.basename(key) + ".jpg"
                image_path_dir = os.path.join(output_dir, base_dir)
                os.makedirs(image_path_dir, exist_ok=True)
                image_path = os.path.join(image_path_dir, filename)

                cv2.imwrite(image_path, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error processing sample {i} from tar {tar_path}: {e}")
                continue


if __name__ == "__main__":
    start_index = sys.argv[1]
    end_index = sys.argv[2]

    tar_url_template = "/ssd/common/datasets/pose-analysis/insta/images/{i:06d}.tar"
    output_dir = "/ssd/common/datasets/pose-analysis/insta/images"

    extract_images(start_index, end_index, tar_url_template, output_dir)