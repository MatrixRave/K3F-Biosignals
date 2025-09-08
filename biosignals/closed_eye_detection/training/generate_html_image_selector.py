from glob import glob
from tqdm import tqdm
import os
##
## A generator for creating a web image selector for data annotation
##

html_header = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Dataset Image Selector (Selectable/Deselectable)</title>
        <style>
    
            body{
                background-color: #000;
            }
    
            img {
                width: 200px;
                height: 200px;
                margin: 3px;
                cursor: pointer;
                object-fit: cover;
                border-radius: 4px;
              
                border: 4px solid transparent;
            }
    
            .selected {
                border-color: green;
            }
        </style>
    </head>
    <body>
    <button onclick="downloadClicked()">Download selected images list</button>
    <div id="imageContainer">
'''

html_footer = '''
    </div>
    <button onclick="downloadClicked()">Download selected images list</button>
    <script>
        const clickedImages = new Set();
    
        document.querySelectorAll("#imageContainer img").forEach(img => {
            img.addEventListener("click", () => {
                const imgName = img.getAttribute('src');
                if (clickedImages.has(imgName)) {
                    clickedImages.delete(imgName);
                    img.classList.remove("selected");
                } else {
                    clickedImages.add(imgName);
                    img.classList.add("selected");
                }
            });
        });
    
        function downloadClicked() {
            if (clickedImages.size === 0) {
                alert("No images selected!");
                return;
            }
            const content = Array.from(clickedImages).join("\\n");
            const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "selected_images_closed_01.txt";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    </script>
    
    </body>
    </html>

'''

def split_into_chunks(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


def generate_html_dataset_selector(html_fn, img_dir, img_suffix='png', split_size=1):
    img_files = glob(os.path.join(img_dir, f"*.{img_suffix}"))
    print(f"Found {len(img_files)} images in {img_dir}")

    img_files_split = split_into_chunks(img_files, split_size)

    for n_batch, batch in enumerate(img_files_split):
        print(f"Batch", n_batch+1)

        if len(img_files_split) > 1:
            out_fp = os.path.join(os.path.dirname(img_dir), html_fn.replace(".html", f"_{n_batch+1}.html"))
        else:
            out_fp = os.path.join(os.path.dirname(img_dir), html_fn)

        with open(out_fp, "w") as fobj:
            fobj.write(html_header)

            for img_file in tqdm(img_files):
                rel_img_fp = os.path.join(os.path.basename(os.path.dirname(img_file)),
                                          os.path.basename(img_file))
                img_html = f'<img src="{rel_img_fp}">'
                fobj.write(img_html)

            fobj.write(html_footer)

        print(out_fp)



if __name__ == "__main__":
    html_fn = "open.html"
    img_dir = "/home/nive1002/mEBAL2-dataset/Blinks-Unblinks_prepared/open"
    img_dir = "Blinks-Unblinks_prepared/open"
    generate_html_dataset_selector(html_fn, img_dir, img_suffix='png', split_size=15)