## Bi-AVOD Experiments

#### AVOD on tracking datasets
`pyramid_cars_with_aug_tracking_results_0.1`
`Iter: 120000`
<table>
   <tr>
      <td style="text-align:center;vertical-align:middle;">Class</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
      <td rowspan="2" style="text-align:center;vertical-align:middle;">Car</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">90.40/90.91</td>
      <td style="text-align:center;vertical-align:middle;">71.71/72.72</td>
      <td style="text-align:center;vertical-align:middle;">71.33/72.72</td>
      <td style="text-align:center;vertical-align:middle;">75.24/90.90</td>
      <td style="text-align:center;vertical-align:middle;">55.11/72.69</td>
      <td style="text-align:center;vertical-align:middle;">48.58/72.66</td>


#### Without correlation module

1. Training from scratch

* $AP_{3D} / AP_{BEV}$ %

`pyramid_cars_with_aug_dt_5_tracking_results_0.1`

`Iter:119000` 

<table>
   <tr>
      <td style="text-align:center;vertical-align:middle;">Class</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
      <td rowspan="2" style="text-align:center;vertical-align:middle;">Car</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">89.58/90.91</td>
      <td style="text-align:center;vertical-align:middle;">79.38/81.79</td>
      <td style="text-align:center;vertical-align:middle;">71.00/72.71</td>
      <td style="text-align:center;vertical-align:middle;">73.58/90.89</td>
      <td style="text-align:center;vertical-align:middle;">54.11/81.75</td>
      <td style="text-align:center;vertical-align:middle;">47.75/72.68</td>
   </tr>
</table>

2. With pretrained model (pretrained in detection datasets)

`pyramid_cars_with_aug_dt_5_tracking_pretrained_results_0.1`

`Iter: 6000`

<table>
   <tr>
      <td style="text-align:center;vertical-align:middle;">Class</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
      <td rowspan="2" style="text-align:center;vertical-align:middle;">Car</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">90.61/99.98</td>
      <td style="text-align:center;vertical-align:middle;">89.01/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.84/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.81/90.91</td>
      <td style="text-align:center;vertical-align:middle;">76.38/90.86</td>
      <td style="text-align:center;vertical-align:middle;">75.83/90.85</td>
   </tr>
</table>


#### With correlation module

1. Training from scratch

`pyramid_cars_with_aug_dt_5_corr_tracking_results_0.1`

`Iter:108000/118000`

<table>
   <tr>
      <td style="text-align:center;vertical-align:middle;">Class</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
      <td rowspan="2" style="text-align:center;vertical-align:middle;">Car</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">90.42/90.91</td>
      <td style="text-align:center;vertical-align:middle;">80.23/81.79</td>
      <td style="text-align:center;vertical-align:middle;">79.88/81.78</td>
      <td style="text-align:center;vertical-align:middle;">76.91/90.89</td>
      <td style="text-align:center;vertical-align:middle;">58.30/81.76</td>
      <td style="text-align:center;vertical-align:middle;">56.91/72.70</td>
   </tr>
</table>




2. With pretrained model (pretrained in detection datasets)

`pyramid_cars_with_aug_dt_5_tracking_corr_pretrained_results_0.1`

`Iter:11000`

<table>
   <tr>
      <td style="text-align:center;vertical-align:middle;">Class</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
      <td rowspan="2" style="text-align:center;vertical-align:middle;">Car</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">90.60/99.94</td>
      <td style="text-align:center;vertical-align:middle;">89.22/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.93/90.89</td>
      <td style="text-align:center;vertical-align:middle;">89.05/90.91</td>
      <td style="text-align:center;vertical-align:middle;">76.67/90.85</td>
      <td style="text-align:center;vertical-align:middle;">75.83/90.84</td>
   </tr>
</table>


### 实验安排

#### 0. 架构对比

* \* means pretrained model

<table>
   <tr>
      <td style="text-align:center;vertical-align:middle;">-</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
       <td style="text-align:center;vertical-align:middle;">Method</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">avod</td>
      <td style="text-align:center;vertical-align:middle;">90.40/90.91</td>
      <td style="text-align:center;vertical-align:middle;">71.71/72.72</td>
      <td style="text-align:center;vertical-align:middle;">71.33/72.72</td>
      <td style="text-align:center;vertical-align:middle;">75.24/90.90</td>
      <td style="text-align:center;vertical-align:middle;">55.11/72.69</td>
      <td style="text-align:center;vertical-align:middle;">48.58/72.66</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">bi-avod </br>(no corr)</td>
      <td style="text-align:center;vertical-align:middle;">89.58/90.91</td>
      <td style="text-align:center;vertical-align:middle;">79.38/81.79</td>
      <td style="text-align:center;vertical-align:middle;">71.00/72.71</td>
      <td style="text-align:center;vertical-align:middle;">73.58/90.89</td>
      <td style="text-align:center;vertical-align:middle;">54.11/81.75</td>
      <td style="text-align:center;vertical-align:middle;">47.75/72.68</td>
    </tr>
	<tr>
      <td style="text-align:center;vertical-align:middle;">bi-avod </br>(with corr)</td>
       <td style="text-align:center;vertical-align:middle;">90.42/90.91</td>
      <td style="text-align:center;vertical-align:middle;">80.23/81.79</td>
      <td style="text-align:center;vertical-align:middle;">79.88/81.78</td>
      <td style="text-align:center;vertical-align:middle;">76.91/90.89</td>
      <td style="text-align:center;vertical-align:middle;">58.30/81.76</td>
      <td style="text-align:center;vertical-align:middle;">56.91/72.70</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">bi-avod* </br>(no corr)</td>
      <td style="text-align:center;vertical-align:middle;">90.61/99.98</td>
      <td style="text-align:center;vertical-align:middle;">89.01/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.84/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.81/90.91</td>
      <td style="text-align:center;vertical-align:middle;">76.38/90.86</td>
      <td style="text-align:center;vertical-align:middle;">75.83/90.85</td>
    </tr>
	<tr>
      <td style="text-align:center;vertical-align:middle;">bi-avod* </br>(with corr)</td>
      <td style="text-align:center;vertical-align:middle;">90.60/99.94</td>
      <td style="text-align:center;vertical-align:middle;">89.22/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.93/90.89</td>
      <td style="text-align:center;vertical-align:middle;">89.05/90.91</td>
      <td style="text-align:center;vertical-align:middle;">76.67/90.85</td>
      <td style="text-align:center;vertical-align:middle;">75.83/90.84</td>
    </tr>
</table>



#### 1. 对 $\lambda$ 的探索

using pretrained_model

不同 $\lambda$ 的检测效果

<table>
   <tr>
      <td colspan="2" style="text-align:center;vertical-align:middle;">-</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
      <td rowspan="8" style="text-align:center;vertical-align:middle;">Car</td>
       <td style="text-align:center;vertical-align:middle;">Stride</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">1</td>
     <td style="text-align:center;vertical-align:middle;">90.60/99.94</td>
      <td style="text-align:center;vertical-align:middle;">89.22/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.93/90.89</td>
      <td style="text-align:center;vertical-align:middle;">89.05/90.91</td>
      <td style="text-align:center;vertical-align:middle;">76.67/90.85</td>
      <td style="text-align:center;vertical-align:middle;">75.83/90.84</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">2</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">3</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">4</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">5</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">6</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">7</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
</table>



#### 2. 多目标追踪 state-of-the-art 对比

<table>
   <tr>
       <td style="text-align:center;vertical-align:middle;">Method</td>
       <td style="text-align:center;vertical-align:middle;">MOTA(%)</td>
      <td style="text-align:center;vertical-align:middle;">MOTP(%)</td>
      <td style="text-align:center;vertical-align:middle;">MT(%)</td>
      <td style="text-align:center;vertical-align:middle;">ML(%)</td>
      <td style="text-align:center;vertical-align:middle;">IDS(%)</td>
      <td style="text-align:center;vertical-align:middle;">FRAG</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">1</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">2</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">3</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">4</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">5</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">6</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">7</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
</table>


#### 3. Correlation 步长对比

 <table>
   <tr>
      <td colspan="2" style="text-align:center;vertical-align:middle;">-</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
      <td rowspan="4" style="text-align:center;vertical-align:middle;">Car</td>
       <td style="text-align:center;vertical-align:middle;">Stride</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
      <td style="text-align:center;vertical-align:middle;">Easy</td>
      <td style="text-align:center;vertical-align:middle;">Moderate</td>
      <td style="text-align:center;vertical-align:middle;">Hard</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">5</td>
     <td style="text-align:center;vertical-align:middle;">90.60/99.94</td>
      <td style="text-align:center;vertical-align:middle;">89.22/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.93/90.89</td>
      <td style="text-align:center;vertical-align:middle;">89.05/90.91</td>
      <td style="text-align:center;vertical-align:middle;">76.67/90.85</td>
      <td style="text-align:center;vertical-align:middle;">75.83/90.84</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">10</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">20</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
</table>



#### 4. Correlation 不同实现对比

① 只对 BEV map 做相关，因为我们只需要得到 (x,z) 的偏移量就行了， 这些信息通过 BEV map 就能很好的包含。

② 不做相关，只是将两 BEV map 变为一维，然后相减/concat，对两帧对应物体框的并区域作为anchor, 提取对应区域特征, 预测特征