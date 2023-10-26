<div align="center">

<h1>PERF: Panoramic Neural Radiance Field from a Single Panorama</h1>

<div>
    Technical Report 2023
</div>

<div>
    <a href='https://wanggcong.github.io/' target='_blank'>Guangcong Wang*<sup>1</sup></a>&emsp;
    <a href='https://quartz-khaan-c6f.notion.site/Peng-Wang-0ab0a2521ecf40f5836581770c14219c' target='_blank'>Peng Wang*<sup>2</sup></a>&emsp;
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen<sup>1</sup></a>&emsp;
    <a href='https://www.cs.hku.hk/people/academic-staff/wenping' target='_blank'>Wenping Wang<sup>2</sup></a>&emsp;
    <a href='https://www.mmlab-ntu.com/person/ccloy/' target='_blank'>Chen Change Loy<sup>1</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>1</sup></a>
</div>
<div>
    S-Lab, Nanyang Technological University<sup>1</sup>, The University of Hong Kong<sup>2</sup>
</div>
<div>
* denotes equal contribution
</div>





### [Project](https://perf-project.github.io/) | [YouTube](https://www.youtube.com/watch?v=4wa2h1fjh2U&feature=youtu.be) | [arXiv](https://arxiv.org/abs/2310.16831) 
<div>

</div>
    
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Totoro97/PeRF)
<!--![visitors](https://visitor-badge.glitch.me/badge?page_id=Totoro97/PeRF)-->
<tr>
    <img src="img/input_output_task_definition3.gif" width="90%"/>
</tr>
</div>




## Usage

### Setup

Step 1: Clone this repository

```
git clone https://github.com/perf-project/PeRF.git
cd PeRF
pip install -r requirements.txt
```

Step 2: Install tiny-cuda-nn

```
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Step 3:
Download checkpoints as shown [here](./pre_checkpoints).

### Train

Here is a command to train a PeRF of an example data:
```
python core_exp_runner.py --config-name nerf dataset.image_path=$(pwd)/example_data/kitchen/image.png device.base_exp_dir=$(pwd)/exp
```


### Render a video

After training is done, you can render a traverse video by running the following command:
```
python core_exp_runner.py --config-name nerf dataset.image_path=$(pwd)/example_data/kitchen/image.png device.base_exp_dir=$(pwd)/exp mode=render_dense is_continue=true
```

## Citation
Cite as below if you find it helpful to your research.

```
@article{perf2023,
    title={PERF: Panoramic Neural Radiance Field from a Single Panorama},
    author={Guangcong Wang and Peng Wang and Zhaoxi Chen and Wenping Wang and Chen Change Loy and Ziwei Liu},
    journal={Technical Report},
    year={2023}}
```
