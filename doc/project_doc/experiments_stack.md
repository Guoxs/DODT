## Stack AVOD Experiments

#### Training from scratch

1. Without integrated features when regression and classification

* $AP_{3D} / AP_{BEV}$ %

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
		<td style="text-align:center;vertical-align:middle;">89.50/90.90</td>
		<td style="text-align:center;vertical-align:middle;">70.16/72.71</td>
		<td style="text-align:center;vertical-align:middle;">70.00/72.70</td>
		<td style="text-align:center;vertical-align:middle;">52.11/90.25</td>
		<td style="text-align:center;vertical-align:middle;">36.58/72.27</td>
		<td style="text-align:center;vertical-align:middle;">36.10/72.30</td>
	</tr>
</table>

2. With integrated features

* $AP_{3D} / AP_{BEV}$ %

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
		<td style="text-align:center;vertical-align:middle;">90.02/90.91</td>
		<td style="text-align:center;vertical-align:middle;">80.18/81.81</td>
		<td style="text-align:center;vertical-align:middle;">79.99/81.80</td>
		<td style="text-align:center;vertical-align:middle;">76.00/90.90</td>
		<td style="text-align:center;vertical-align:middle;">57.23/81.74</td>
		<td style="text-align:center;vertical-align:middle;">56.13/72.70</td>
	</tr>
</table>


#### With pretrained model


1. Without integrated features when regression and classification

* $AP_{3D} / AP_{BEV}$ %

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
		<td style="text-align:center;vertical-align:middle;">98.45/99.99</td>
		<td style="text-align:center;vertical-align:middle;">89.00/90.90</td>
		<td style="text-align:center;vertical-align:middle;">88.74/90.89</td>
		<td style="text-align:center;vertical-align:middle;">88.82/99.97</td>
		<td style="text-align:center;vertical-align:middle;">76.07/90.88</td>
		<td style="text-align:center;vertical-align:middle;">75.45/90.86</td>
	</tr>
</table>

2. With integrated features

* $AP_{3D} / AP_{BEV}$ %

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
		<td style="text-align:center;vertical-align:middle;"> placeholder </td>
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
       <td style="text-align:center;vertical-align:middle;">Recall(%)</td>
       <td style="text-align:center;vertical-align:middle;">Precision(%)</td>
   </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">1</td>
      <td style="text-align:center;vertical-align:middle;">76.73</td>
      <td style="text-align:center;vertical-align:middle;">84.47</td>
      <td style="text-align:center;vertical-align:middle;">65.55</td>
      <td style="text-align:center;vertical-align:middle;">6.30</td>
      <td style="text-align:center;vertical-align:middle;">15</td>
      <td style="text-align:center;vertical-align:middle;">121</td>
      <td style="text-align:center;vertical-align:middle;">80.57</td>
      <td style="text-align:center;vertical-align:middle;">97.73</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">-1-</td>
      <td style="text-align:center;vertical-align:middle;">78.49</td>
      <td style="text-align:center;vertical-align:middle;">83.82</td>
      <td style="text-align:center;vertical-align:middle;">70.59</td>
      <td style="text-align:center;vertical-align:middle;">5.04</td>
      <td style="text-align:center;vertical-align:middle;">63</td>
      <td style="text-align:center;vertical-align:middle;">155</td>
      <td style="text-align:center;vertical-align:middle;">84.45</td>
      <td style="text-align:center;vertical-align:middle;">95.94</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">+1+</td>
      <td style="text-align:center;vertical-align:middle;">77.57</td>
      <td style="text-align:center;vertical-align:middle;">83.60</td>
      <td style="text-align:center;vertical-align:middle;">73.11</td>
      <td style="text-align:center;vertical-align:middle;">5.04</td>
      <td style="text-align:center;vertical-align:middle;">68</td>
      <td style="text-align:center;vertical-align:middle;">156</td>
      <td style="text-align:center;vertical-align:middle;">85.52</td>
      <td style="text-align:center;vertical-align:middle;">94.15</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">-2-</td>
      <td style="text-align:center;vertical-align:middle;">77.82</td>
      <td style="text-align:center;vertical-align:middle;">84.77</td>
      <td style="text-align:center;vertical-align:middle;">65.97</td>
      <td style="text-align:center;vertical-align:middle;">7.14</td>
      <td style="text-align:center;vertical-align:middle;">14</td>
      <td style="text-align:center;vertical-align:middle;">93</td>
      <td style="text-align:center;vertical-align:middle;">81.95</td>
      <td style="text-align:center;vertical-align:middle;">97.19</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">-2+</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">+2-</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">+2+</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
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
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
</table>



#### 3. 物体检测指标

<table>
   <tr>
      <td colspan="2" style="text-align:center;vertical-align:middle;">-</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.5</td>
      <td colspan="3" style="text-align:center;vertical-align:middle;">IOU = 0.7</td>
   <tr>
   <tr>
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
      <td style="text-align:center;vertical-align:middle;">90.59/99.98</td>
      <td style="text-align:center;vertical-align:middle;">88.94/90.90</td>
      <td style="text-align:center;vertical-align:middle;">88.76/90.89</td>
      <td style="text-align:center;vertical-align:middle;">88.33/90.90</td>
      <td style="text-align:center;vertical-align:middle;">75.25/90.86</td>
      <td style="text-align:center;vertical-align:middle;">68.51/90.85</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">-1-</td>
      <td style="text-align:center;vertical-align:middle;">85.87/99.98</td>
      <td style="text-align:center;vertical-align:middle;">83.88/90.88</td>
      <td style="text-align:center;vertical-align:middle;">84.32/90.88</td>
      <td style="text-align:center;vertical-align:middle;">78.53/90.82</td>
      <td style="text-align:center;vertical-align:middle;">64.84/90.77</td>
      <td style="text-align:center;vertical-align:middle;">59.36/90.78</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">+1+</td>
      <td style="text-align:center;vertical-align:middle;">83.60/99.90</td>
      <td style="text-align:center;vertical-align:middle;">81.89/90.83</td>
      <td style="text-align:center;vertical-align:middle;">82.53/90.84</td>
      <td style="text-align:center;vertical-align:middle;">76.05/90.76</td>
      <td style="text-align:center;vertical-align:middle;">63.23/90.69</td>
      <td style="text-align:center;vertical-align:middle;">58.04/90.71</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">-2-</td>
      <td style="text-align:center;vertical-align:middle;">88.58/90.91</td>
      <td style="text-align:center;vertical-align:middle;">86.21/90.89</td>
      <td style="text-align:center;vertical-align:middle;">86.54/90.89</td>
      <td style="text-align:center;vertical-align:middle;">84.00/90.83</td>
      <td style="text-align:center;vertical-align:middle;">70.02/90.82</td>
      <td style="text-align:center;vertical-align:middle;">70.03/81.76</td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">-2+</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">+2-</td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
      <td style="text-align:center;vertical-align:middle;"> </td>
    </tr>
    <tr>
      <td style="text-align:center;vertical-align:middle;">+2+</td>
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
</table>