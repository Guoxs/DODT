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

