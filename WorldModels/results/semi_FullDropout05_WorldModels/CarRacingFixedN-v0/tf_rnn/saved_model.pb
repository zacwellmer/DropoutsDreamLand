??+
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??*
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
dropout_lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#?*.
shared_namedropout_lstm/lstm_cell/kernel
?
1dropout_lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpdropout_lstm/lstm_cell/kernel*
_output_shapes
:	#?*
dtype0
?
'dropout_lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*8
shared_name)'dropout_lstm/lstm_cell/recurrent_kernel
?
;dropout_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp'dropout_lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
dropout_lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namedropout_lstm/lstm_cell/bias
?
/dropout_lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpdropout_lstm/lstm_cell/bias*
_output_shapes	
:?*
dtype0
?
mu_logstd_logmix_net/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namemu_logstd_logmix_net/kernel
?
/mu_logstd_logmix_net/kernel/Read/ReadVariableOpReadVariableOpmu_logstd_logmix_net/kernel* 
_output_shapes
:
??*
dtype0
?
mu_logstd_logmix_net/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namemu_logstd_logmix_net/bias
?
-mu_logstd_logmix_net/bias/Read/ReadVariableOpReadVariableOpmu_logstd_logmix_net/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
?
$Adam/dropout_lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#?*5
shared_name&$Adam/dropout_lstm/lstm_cell/kernel/m
?
8Adam/dropout_lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/dropout_lstm/lstm_cell/kernel/m*
_output_shapes
:	#?*
dtype0
?
.Adam/dropout_lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*?
shared_name0.Adam/dropout_lstm/lstm_cell/recurrent_kernel/m
?
BAdam/dropout_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp.Adam/dropout_lstm/lstm_cell/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
"Adam/dropout_lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/dropout_lstm/lstm_cell/bias/m
?
6Adam/dropout_lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOp"Adam/dropout_lstm/lstm_cell/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/mu_logstd_logmix_net/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/mu_logstd_logmix_net/kernel/m
?
6Adam/mu_logstd_logmix_net/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/mu_logstd_logmix_net/kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/mu_logstd_logmix_net/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/mu_logstd_logmix_net/bias/m
?
4Adam/mu_logstd_logmix_net/bias/m/Read/ReadVariableOpReadVariableOp Adam/mu_logstd_logmix_net/bias/m*
_output_shapes	
:?*
dtype0
?
$Adam/dropout_lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#?*5
shared_name&$Adam/dropout_lstm/lstm_cell/kernel/v
?
8Adam/dropout_lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/dropout_lstm/lstm_cell/kernel/v*
_output_shapes
:	#?*
dtype0
?
.Adam/dropout_lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*?
shared_name0.Adam/dropout_lstm/lstm_cell/recurrent_kernel/v
?
BAdam/dropout_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp.Adam/dropout_lstm/lstm_cell/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
"Adam/dropout_lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/dropout_lstm/lstm_cell/bias/v
?
6Adam/dropout_lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOp"Adam/dropout_lstm/lstm_cell/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/mu_logstd_logmix_net/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/mu_logstd_logmix_net/kernel/v
?
6Adam/mu_logstd_logmix_net/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/mu_logstd_logmix_net/kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/mu_logstd_logmix_net/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/mu_logstd_logmix_net/bias/v
?
4Adam/mu_logstd_logmix_net/bias/v/Read/ReadVariableOpReadVariableOp Adam/mu_logstd_logmix_net/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?)
value?)B?) B?)
?
	optimizer
loss_fn
inference_base
out_net
loss
trainable_variables

signatures
	keras_api
		variables

regularization_losses
?
iter

beta_1

beta_2
	decay
learning_ratemVmWmXmYmZv[v\v]v^v_
 
l
cell

state_spec
trainable_variables
	keras_api
	variables
regularization_losses
y
layer_with_weights-0
layer-0
trainable_variables
	keras_api
	variables
regularization_losses
 
#
0
1
2
3
4
 
?
 layer_metrics
trainable_variables
		variables
!non_trainable_variables
"metrics
#layer_regularization_losses

regularization_losses

$layers
#
0
1
2
3
4
 
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
~

kernel
recurrent_kernel
bias
%trainable_variables
&	keras_api
'	variables
(regularization_losses
 

0
1
2
?
)layer_metrics
trainable_variables
	variables
*non_trainable_variables
+metrics
,layer_regularization_losses
regularization_losses

-states

.layers

0
1
2
 
h

kernel
bias
/trainable_variables
0	keras_api
1	variables
2regularization_losses

0
1
?
3layer_metrics
trainable_variables
	variables
4non_trainable_variables
5metrics
6layer_regularization_losses
regularization_losses

7layers

0
1
 
ca
VARIABLE_VALUEdropout_lstm/lstm_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'dropout_lstm/lstm_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEdropout_lstm/lstm_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEmu_logstd_logmix_net/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEmu_logstd_logmix_net/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
 
 

80
91
:2
;3
 

0
1

0
1
2
?
<layer_metrics
%trainable_variables
'	variables
=non_trainable_variables
>metrics
?layer_regularization_losses
(regularization_losses

@layers

0
1
2
 
 
 
 
 
 

0

0
1
?
Alayer_metrics
/trainable_variables
1	variables
Bnon_trainable_variables
Cmetrics
Dlayer_regularization_losses
2regularization_losses

Elayers

0
1
 
 
 
 
 

0
4
	Ftotal
	Gcount
H	keras_api
I	variables
4
	Jtotal
	Kcount
L	keras_api
M	variables
4
	Ntotal
	Ocount
P	keras_api
Q	variables
4
	Rtotal
	Scount
T	keras_api
U	variables
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

I	variables

F0
G1
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

M	variables

J0
K1
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

Q	variables

N0
O1
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

U	variables

R0
S1
??
VARIABLE_VALUE$Adam/dropout_lstm/lstm_cell/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/dropout_lstm/lstm_cell/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/dropout_lstm/lstm_cell/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/mu_logstd_logmix_net/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/mu_logstd_logmix_net/bias/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/dropout_lstm/lstm_cell/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/dropout_lstm/lstm_cell/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/dropout_lstm/lstm_cell/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/mu_logstd_logmix_net/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/mu_logstd_logmix_net/bias/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*,
_output_shapes
:??????????#*
dtype0*!
shape:??????????#
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dropout_lstm/lstm_cell/kerneldropout_lstm/lstm_cell/bias'dropout_lstm/lstm_cell/recurrent_kernelmu_logstd_logmix_net/kernelmu_logstd_logmix_net/bias*
Tin

2*
Tout
2*N
_output_shapes<
::??????????:?????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*.
f)R'
%__inference_signature_wrapper_6426219
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp1dropout_lstm/lstm_cell/kernel/Read/ReadVariableOp;dropout_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp/dropout_lstm/lstm_cell/bias/Read/ReadVariableOp/mu_logstd_logmix_net/kernel/Read/ReadVariableOp-mu_logstd_logmix_net/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp8Adam/dropout_lstm/lstm_cell/kernel/m/Read/ReadVariableOpBAdam/dropout_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp6Adam/dropout_lstm/lstm_cell/bias/m/Read/ReadVariableOp6Adam/mu_logstd_logmix_net/kernel/m/Read/ReadVariableOp4Adam/mu_logstd_logmix_net/bias/m/Read/ReadVariableOp8Adam/dropout_lstm/lstm_cell/kernel/v/Read/ReadVariableOpBAdam/dropout_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp6Adam/dropout_lstm/lstm_cell/bias/v/Read/ReadVariableOp6Adam/mu_logstd_logmix_net/kernel/v/Read/ReadVariableOp4Adam/mu_logstd_logmix_net/bias/v/Read/ReadVariableOpConst*)
Tin"
 2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*)
f$R"
 __inference__traced_save_6429173
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedropout_lstm/lstm_cell/kernel'dropout_lstm/lstm_cell/recurrent_kerneldropout_lstm/lstm_cell/biasmu_logstd_logmix_net/kernelmu_logstd_logmix_net/biastotalcounttotal_1count_1total_2count_2total_3count_3$Adam/dropout_lstm/lstm_cell/kernel/m.Adam/dropout_lstm/lstm_cell/recurrent_kernel/m"Adam/dropout_lstm/lstm_cell/bias/m"Adam/mu_logstd_logmix_net/kernel/m Adam/mu_logstd_logmix_net/bias/m$Adam/dropout_lstm/lstm_cell/kernel/v.Adam/dropout_lstm/lstm_cell/recurrent_kernel/v"Adam/dropout_lstm/lstm_cell/bias/v"Adam/mu_logstd_logmix_net/kernel/v Adam/mu_logstd_logmix_net/bias/v*(
Tin!
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*,
f'R%
#__inference__traced_restore_6429269??)
??
?
dropout_lstm_while_body_6426372#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
dropout_lstm_strided_slice_1_0^
Ztensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5 
dropout_lstm_strided_slice_1\
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162#
!lstm_cell/StatefulPartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yk
add_1AddV2dropout_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identity%dropout_lstm_while_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identitylstm_cell/mul_10:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_4?

Identity_5Identitylstm_cell/add_3:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5">
dropout_lstm_strided_slice_1dropout_lstm_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"?
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensorZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6425707

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity

identity_1

identity_2??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????#2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162#
!lstm_cell/StatefulPartitionedCallt
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_6425536*
condR
while_cond_6425535*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*-
_output_shapes
:???????????2

Identity?

Identity_1Identitywhile:output:4"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identitywhile:output:5"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????#:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_6425080
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_6425080___redundant_placeholder0/
+while_cond_6425080___redundant_placeholder1/
+while_cond_6425080___redundant_placeholder2/
+while_cond_6425080___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?p
?
dropout_lstm_while_body_6426684#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
dropout_lstm_strided_slice_1_0^
Ztensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5 
dropout_lstm_strided_slice_1\
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/PartitionedCallPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yk
add_1AddV2dropout_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityl

Identity_1Identity%dropout_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_5">
dropout_lstm_strided_slice_1dropout_lstm_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"?
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensorZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
+__inference_lstm_cell_layer_call_fn_6428822

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_64247032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????#:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?n
?
while_body_6425827
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/PartitionedCallPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?	
?
G__inference_sequential_layer_call_and_return_conditional_losses_6425343
input_1 
mu_logstd_logmix_net_6425337 
mu_logstd_logmix_net_6425339
identity??,mu_logstd_logmix_net/StatefulPartitionedCall?
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinput_1mu_logstd_logmix_net_6425337mu_logstd_logmix_net_6425339*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_64253172.
,mu_logstd_logmix_net/StatefulPartitionedCall?
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0-^mu_logstd_logmix_net/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_dropout_lstm_layer_call_fn_6428752
inputs_0
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*]
_output_shapesK
I:???????????????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_64251522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????#:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????#
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6427225
input_18
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1

identity_2??.dropout_lstm/lstm_cell/StatefulPartitionedCall?dropout_lstm/while}
dropout_lstm/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:??????????#2
dropout_lstm/Castm
dropout_lstm/ShapeShapedropout_lstm/Cast:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape?
 dropout_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dropout_lstm/strided_slice/stack?
"dropout_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_1?
"dropout_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_2?
dropout_lstm/strided_sliceStridedSlicedropout_lstm/Shape:output:0)dropout_lstm/strided_slice/stack:output:0+dropout_lstm/strided_slice/stack_1:output:0+dropout_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slicew
dropout_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/mul/y?
dropout_lstm/zeros/mulMul#dropout_lstm/strided_slice:output:0!dropout_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/muly
dropout_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/Less/y?
dropout_lstm/zeros/LessLessdropout_lstm/zeros/mul:z:0"dropout_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/Less}
dropout_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/packed/1?
dropout_lstm/zeros/packedPack#dropout_lstm/strided_slice:output:0$dropout_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros/packedy
dropout_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros/Const?
dropout_lstm/zerosFill"dropout_lstm/zeros/packed:output:0!dropout_lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/mul/y?
dropout_lstm/zeros_1/mulMul#dropout_lstm/strided_slice:output:0#dropout_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/mul}
dropout_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/Less/y?
dropout_lstm/zeros_1/LessLessdropout_lstm/zeros_1/mul:z:0$dropout_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/Less?
dropout_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/packed/1?
dropout_lstm/zeros_1/packedPack#dropout_lstm/strided_slice:output:0&dropout_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros_1/packed}
dropout_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros_1/Const?
dropout_lstm/zeros_1Fill$dropout_lstm/zeros_1/packed:output:0#dropout_lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros_1?
dropout_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose/perm?
dropout_lstm/transpose	Transposedropout_lstm/Cast:y:0$dropout_lstm/transpose/perm:output:0*
T0*,
_output_shapes
:??????????#2
dropout_lstm/transposev
dropout_lstm/Shape_1Shapedropout_lstm/transpose:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape_1?
"dropout_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_1/stack?
$dropout_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_1?
$dropout_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_2?
dropout_lstm/strided_slice_1StridedSlicedropout_lstm/Shape_1:output:0+dropout_lstm/strided_slice_1/stack:output:0-dropout_lstm/strided_slice_1/stack_1:output:0-dropout_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slice_1?
(dropout_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(dropout_lstm/TensorArrayV2/element_shape?
dropout_lstm/TensorArrayV2TensorListReserve1dropout_lstm/TensorArrayV2/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2?
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   2D
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
4dropout_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordropout_lstm/transpose:y:0Kdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4dropout_lstm/TensorArrayUnstack/TensorListFromTensor?
"dropout_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_2/stack?
$dropout_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_1?
$dropout_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_2?
dropout_lstm/strided_slice_2StridedSlicedropout_lstm/transpose:y:0+dropout_lstm/strided_slice_2/stack:output:0-dropout_lstm/strided_slice_2/stack_1:output:0-dropout_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
.dropout_lstm/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall%dropout_lstm/strided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_13061620
.dropout_lstm/lstm_cell/StatefulPartitionedCall?
&dropout_lstm/lstm_cell/ones_like/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/ones_like/Shape?
&dropout_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&dropout_lstm/lstm_cell/ones_like/Const?
 dropout_lstm/lstm_cell/ones_likeFill/dropout_lstm/lstm_cell/ones_like/Shape:output:0/dropout_lstm/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/ones_like?
$dropout_lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2&
$dropout_lstm/lstm_cell/dropout/Const?
"dropout_lstm/lstm_cell/dropout/MulMul)dropout_lstm/lstm_cell/ones_like:output:0-dropout_lstm/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2$
"dropout_lstm/lstm_cell/dropout/Mul?
$dropout_lstm/lstm_cell/dropout/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$dropout_lstm/lstm_cell/dropout/Shape?
;dropout_lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform-dropout_lstm/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2=
;dropout_lstm/lstm_cell/dropout/random_uniform/RandomUniform?
-dropout_lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2/
-dropout_lstm/lstm_cell/dropout/GreaterEqual/y?
+dropout_lstm/lstm_cell/dropout/GreaterEqualGreaterEqualDdropout_lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:06dropout_lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2-
+dropout_lstm/lstm_cell/dropout/GreaterEqual?
#dropout_lstm/lstm_cell/dropout/CastCast/dropout_lstm/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2%
#dropout_lstm/lstm_cell/dropout/Cast?
$dropout_lstm/lstm_cell/dropout/Mul_1Mul&dropout_lstm/lstm_cell/dropout/Mul:z:0'dropout_lstm/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout/Mul_1?
&dropout_lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_1/Const?
$dropout_lstm/lstm_cell/dropout_1/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_1/Mul?
&dropout_lstm/lstm_cell/dropout_1/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_1/Shape?
=dropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_1/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_1/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_1/CastCast1dropout_lstm/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_1/Cast?
&dropout_lstm/lstm_cell/dropout_1/Mul_1Mul(dropout_lstm/lstm_cell/dropout_1/Mul:z:0)dropout_lstm/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_1/Mul_1?
&dropout_lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_2/Const?
$dropout_lstm/lstm_cell/dropout_2/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_2/Mul?
&dropout_lstm/lstm_cell/dropout_2/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_2/Shape?
=dropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2܆2?
=dropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_2/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_2/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_2/CastCast1dropout_lstm/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_2/Cast?
&dropout_lstm/lstm_cell/dropout_2/Mul_1Mul(dropout_lstm/lstm_cell/dropout_2/Mul:z:0)dropout_lstm/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_2/Mul_1?
&dropout_lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_3/Const?
$dropout_lstm/lstm_cell/dropout_3/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_3/Mul?
&dropout_lstm/lstm_cell/dropout_3/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_3/Shape?
=dropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??2?
=dropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_3/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_3/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_3/CastCast1dropout_lstm/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_3/Cast?
&dropout_lstm/lstm_cell/dropout_3/Mul_1Mul(dropout_lstm/lstm_cell/dropout_3/Mul:z:0)dropout_lstm/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_3/Mul_1?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:07dropout_lstm/lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:07dropout_lstm/lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:07dropout_lstm/lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:07dropout_lstm/lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_3~
dropout_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
dropout_lstm/lstm_cell/Const?
&dropout_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&dropout_lstm/lstm_cell/split/split_dim?
+dropout_lstm/lstm_cell/split/ReadVariableOpReadVariableOp4dropout_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_3?
dropout_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout_lstm/lstm_cell/Const_1?
(dropout_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(dropout_lstm/lstm_cell/split_1/split_dim?
-dropout_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp6dropout_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0(dropout_lstm/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
*dropout_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*dropout_lstm/lstm_cell/strided_slice/stack?
,dropout_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$dropout_lstm/lstm_cell/strided_slice?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0-dropout_lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_3/stack?
.dropout_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_3/stack_1?
.dropout_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_3/stack_2?
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
*dropout_lstm/TensorArrayV2_1/element_shape?
dropout_lstm/TensorArrayV2_1TensorListReserve3dropout_lstm/TensorArrayV2_1/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2_1h
dropout_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_lstm/time?
%dropout_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%dropout_lstm/while/maximum_iterations?
dropout_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
dropout_lstm/while/loop_counter?
dropout_lstm/whileWhile(dropout_lstm/while/loop_counter:output:0.dropout_lstm/while/maximum_iterations:output:0dropout_lstm/time:output:0%dropout_lstm/TensorArrayV2_1:handle:0dropout_lstm/zeros:output:0dropout_lstm/zeros_1:output:0%dropout_lstm/strided_slice_1:output:0Ddropout_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:04dropout_lstm_lstm_cell_split_readvariableop_resource6dropout_lstm_lstm_cell_split_1_readvariableop_resource.dropout_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_6427034*+
cond#R!
dropout_lstm_while_cond_6427033*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype021
/dropout_lstm/TensorArrayV2Stack/TensorListStack?
"dropout_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"dropout_lstm/strided_slice_3/stack?
$dropout_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$dropout_lstm/strided_slice_3/stack_1?
$dropout_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_3/stack_2?
dropout_lstm/strided_slice_3StridedSlice8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+dropout_lstm/strided_slice_3/stack:output:0-dropout_lstm/strided_slice_3/stack_1:output:0-dropout_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
dropout_lstm/strided_slice_3?
dropout_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose_1/perm?
dropout_lstm/transpose_1	Transpose8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&dropout_lstm/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
dropout_lstm/transpose_1?
dropout_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/runtimeo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential/mu_logstd_logmix_net/BiasAdd{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2?
IdentityIdentitystrided_slice:output:0/^dropout_lstm/lstm_cell/StatefulPartitionedCall^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_2:output:0/^dropout_lstm/lstm_cell/StatefulPartitionedCall^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitystrided_slice_1:output:0/^dropout_lstm/lstm_cell/StatefulPartitionedCall^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::2`
.dropout_lstm/lstm_cell/StatefulPartitionedCall.dropout_lstm/lstm_cell/StatefulPartitionedCall2(
dropout_lstm/whiledropout_lstm/while:U Q
,
_output_shapes
:??????????#
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
dropout_lstm_while_cond_6426683#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_6426683___redundant_placeholder0<
8dropout_lstm_while_cond_6426683___redundant_placeholder1<
8dropout_lstm_while_cond_6426683___redundant_placeholder2<
8dropout_lstm_while_cond_6426683___redundant_placeholder3
identity
e
LessLessplaceholder!less_dropout_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
??
?
"__inference__wrapped_model_6424541
input_18
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1

identity_2??dropout_lstm/while}
dropout_lstm/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:??????????#2
dropout_lstm/Castm
dropout_lstm/ShapeShapedropout_lstm/Cast:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape?
 dropout_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dropout_lstm/strided_slice/stack?
"dropout_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_1?
"dropout_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_2?
dropout_lstm/strided_sliceStridedSlicedropout_lstm/Shape:output:0)dropout_lstm/strided_slice/stack:output:0+dropout_lstm/strided_slice/stack_1:output:0+dropout_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slicew
dropout_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/mul/y?
dropout_lstm/zeros/mulMul#dropout_lstm/strided_slice:output:0!dropout_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/muly
dropout_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/Less/y?
dropout_lstm/zeros/LessLessdropout_lstm/zeros/mul:z:0"dropout_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/Less}
dropout_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/packed/1?
dropout_lstm/zeros/packedPack#dropout_lstm/strided_slice:output:0$dropout_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros/packedy
dropout_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros/Const?
dropout_lstm/zerosFill"dropout_lstm/zeros/packed:output:0!dropout_lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/mul/y?
dropout_lstm/zeros_1/mulMul#dropout_lstm/strided_slice:output:0#dropout_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/mul}
dropout_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/Less/y?
dropout_lstm/zeros_1/LessLessdropout_lstm/zeros_1/mul:z:0$dropout_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/Less?
dropout_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/packed/1?
dropout_lstm/zeros_1/packedPack#dropout_lstm/strided_slice:output:0&dropout_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros_1/packed}
dropout_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros_1/Const?
dropout_lstm/zeros_1Fill$dropout_lstm/zeros_1/packed:output:0#dropout_lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros_1?
dropout_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose/perm?
dropout_lstm/transpose	Transposedropout_lstm/Cast:y:0$dropout_lstm/transpose/perm:output:0*
T0*,
_output_shapes
:??????????#2
dropout_lstm/transposev
dropout_lstm/Shape_1Shapedropout_lstm/transpose:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape_1?
"dropout_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_1/stack?
$dropout_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_1?
$dropout_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_2?
dropout_lstm/strided_slice_1StridedSlicedropout_lstm/Shape_1:output:0+dropout_lstm/strided_slice_1/stack:output:0-dropout_lstm/strided_slice_1/stack_1:output:0-dropout_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slice_1?
(dropout_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(dropout_lstm/TensorArrayV2/element_shape?
dropout_lstm/TensorArrayV2TensorListReserve1dropout_lstm/TensorArrayV2/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2?
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   2D
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
4dropout_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordropout_lstm/transpose:y:0Kdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4dropout_lstm/TensorArrayUnstack/TensorListFromTensor?
"dropout_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_2/stack?
$dropout_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_1?
$dropout_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_2?
dropout_lstm/strided_slice_2StridedSlicedropout_lstm/transpose:y:0+dropout_lstm/strided_slice_2/stack:output:0-dropout_lstm/strided_slice_2/stack_1:output:0-dropout_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
&dropout_lstm/lstm_cell/PartitionedCallPartitionedCall%dropout_lstm/strided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442(
&dropout_lstm/lstm_cell/PartitionedCall?
&dropout_lstm/lstm_cell/ones_like/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/ones_like/Shape?
&dropout_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&dropout_lstm/lstm_cell/ones_like/Const?
 dropout_lstm/lstm_cell/ones_likeFill/dropout_lstm/lstm_cell/ones_like/Shape:output:0/dropout_lstm/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/ones_like?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_3~
dropout_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
dropout_lstm/lstm_cell/Const?
&dropout_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&dropout_lstm/lstm_cell/split/split_dim?
+dropout_lstm/lstm_cell/split/ReadVariableOpReadVariableOp4dropout_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_3?
dropout_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout_lstm/lstm_cell/Const_1?
(dropout_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(dropout_lstm/lstm_cell/split_1/split_dim?
-dropout_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp6dropout_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
*dropout_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*dropout_lstm/lstm_cell/strided_slice/stack?
,dropout_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$dropout_lstm/lstm_cell/strided_slice?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0-dropout_lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_3/stack?
.dropout_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_3/stack_1?
.dropout_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_3/stack_2?
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
*dropout_lstm/TensorArrayV2_1/element_shape?
dropout_lstm/TensorArrayV2_1TensorListReserve3dropout_lstm/TensorArrayV2_1/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2_1h
dropout_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_lstm/time?
%dropout_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%dropout_lstm/while/maximum_iterations?
dropout_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
dropout_lstm/while/loop_counter?
dropout_lstm/whileWhile(dropout_lstm/while/loop_counter:output:0.dropout_lstm/while/maximum_iterations:output:0dropout_lstm/time:output:0%dropout_lstm/TensorArrayV2_1:handle:0dropout_lstm/zeros:output:0dropout_lstm/zeros_1:output:0%dropout_lstm/strided_slice_1:output:0Ddropout_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:04dropout_lstm_lstm_cell_split_readvariableop_resource6dropout_lstm_lstm_cell_split_1_readvariableop_resource.dropout_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_6424382*+
cond#R!
dropout_lstm_while_cond_6424381*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype021
/dropout_lstm/TensorArrayV2Stack/TensorListStack?
"dropout_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"dropout_lstm/strided_slice_3/stack?
$dropout_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$dropout_lstm/strided_slice_3/stack_1?
$dropout_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_3/stack_2?
dropout_lstm/strided_slice_3StridedSlice8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+dropout_lstm/strided_slice_3/stack:output:0-dropout_lstm/strided_slice_3/stack_1:output:0-dropout_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
dropout_lstm/strided_slice_3?
dropout_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose_1/perm?
dropout_lstm/transpose_1	Transpose8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&dropout_lstm/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
dropout_lstm/transpose_1?
dropout_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/runtimeo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential/mu_logstd_logmix_net/BiasAdd{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2?
IdentityIdentitystrided_slice:output:0^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_2:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitystrided_slice_1:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::2(
dropout_lstm/whiledropout_lstm/while:U Q
,
_output_shapes
:??????????#
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?n
?
while_body_6428598
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/PartitionedCallPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?H
?
 __inference__traced_save_6429173
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop<
8savev2_dropout_lstm_lstm_cell_kernel_read_readvariableopF
Bsavev2_dropout_lstm_lstm_cell_recurrent_kernel_read_readvariableop:
6savev2_dropout_lstm_lstm_cell_bias_read_readvariableop:
6savev2_mu_logstd_logmix_net_kernel_read_readvariableop8
4savev2_mu_logstd_logmix_net_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableopC
?savev2_adam_dropout_lstm_lstm_cell_kernel_m_read_readvariableopM
Isavev2_adam_dropout_lstm_lstm_cell_recurrent_kernel_m_read_readvariableopA
=savev2_adam_dropout_lstm_lstm_cell_bias_m_read_readvariableopA
=savev2_adam_mu_logstd_logmix_net_kernel_m_read_readvariableop?
;savev2_adam_mu_logstd_logmix_net_bias_m_read_readvariableopC
?savev2_adam_dropout_lstm_lstm_cell_kernel_v_read_readvariableopM
Isavev2_adam_dropout_lstm_lstm_cell_recurrent_kernel_v_read_readvariableopA
=savev2_adam_dropout_lstm_lstm_cell_bias_v_read_readvariableopA
=savev2_adam_mu_logstd_logmix_net_kernel_v_read_readvariableop?
;savev2_adam_mu_logstd_logmix_net_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ee2bceb5a33a4552b06ef8cca24db21f/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop8savev2_dropout_lstm_lstm_cell_kernel_read_readvariableopBsavev2_dropout_lstm_lstm_cell_recurrent_kernel_read_readvariableop6savev2_dropout_lstm_lstm_cell_bias_read_readvariableop6savev2_mu_logstd_logmix_net_kernel_read_readvariableop4savev2_mu_logstd_logmix_net_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop?savev2_adam_dropout_lstm_lstm_cell_kernel_m_read_readvariableopIsavev2_adam_dropout_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop=savev2_adam_dropout_lstm_lstm_cell_bias_m_read_readvariableop=savev2_adam_mu_logstd_logmix_net_kernel_m_read_readvariableop;savev2_adam_mu_logstd_logmix_net_bias_m_read_readvariableop?savev2_adam_dropout_lstm_lstm_cell_kernel_v_read_readvariableopIsavev2_adam_dropout_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop=savev2_adam_dropout_lstm_lstm_cell_bias_v_read_readvariableop=savev2_adam_mu_logstd_logmix_net_kernel_v_read_readvariableop;savev2_adam_mu_logstd_logmix_net_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	#?:
??:?:
??:?: : : : : : : : :	#?:
??:?:
??:?:	#?:
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	#?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	#?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	#?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428125

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity

identity_1

identity_2??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????#2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
strided_slice_2?
lstm_cell/PartitionedCallPartitionedCallstrided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCallt
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_6427986*
condR
while_cond_6427985*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeq
IdentityIdentitytranspose_1:y:0^while*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????#:::2
whilewhile:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_dropout_lstm_layer_call_fn_6428767
inputs_0
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*]
_output_shapesK
I:???????????????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_64252902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????#:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????#
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
G__inference_sequential_layer_call_and_return_conditional_losses_6428787

inputs7
3mu_logstd_logmix_net_matmul_readvariableop_resource8
4mu_logstd_logmix_net_biasadd_readvariableop_resource
identity??
*mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp3mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*mu_logstd_logmix_net/MatMul/ReadVariableOp?
mu_logstd_logmix_net/MatMulMatMulinputs2mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mu_logstd_logmix_net/MatMul?
+mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp4mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
mu_logstd_logmix_net/BiasAddBiasAdd%mu_logstd_logmix_net/MatMul:product:03mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mu_logstd_logmix_net/BiasAddz
IdentityIdentity%mu_logstd_logmix_net/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
%__inference_signature_wrapper_6426219
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*N
_output_shapes<
::??????????:?????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*+
f&R$
"__inference__wrapped_model_64245412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????#
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_6425317

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_6429051

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_6427694
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_6427694___redundant_placeholder0/
+while_cond_6427694___redundant_placeholder1/
+while_cond_6427694___redundant_placeholder2/
+while_cond_6427694___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?Y
v
'__inference__create_dropout_mask_134875

inputs
identity

identity_1

identity_2

identity_3?{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
ellipsis_mask2
strided_sliceh
ones_like/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:????????? 2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??72(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_3/Mul_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_1n
ones_like_1/ShapeShapestrided_slice_1:output:0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_2n
ones_like_2/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_2/Const?
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_3n
ones_like_3/ShapeShapestrided_slice_3:output:0*
T0*
_output_shapes
:2
ones_like_3/Shapek
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_3/Const?
ones_like_3Fillones_like_3/Shape:output:0ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_4n
ones_like_4/ShapeShapestrided_slice_4:output:0*
T0*
_output_shapes
:2
ones_like_4/Shapek
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_4/Const?
ones_like_4Fillones_like_4/Shape:output:0ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2dropout/Mul_1:z:0ones_like_1:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/axis?
concat_1ConcatV2dropout_1/Mul_1:z:0ones_like_2:output:0concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_1i
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV2dropout_2/Mul_1:z:0ones_like_3:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_2i
concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_3/axis?
concat_3ConcatV2dropout_3/Mul_1:z:0ones_like_4:output:0concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_3c
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????#2

Identityi

Identity_1Identityconcat_1:output:0*
T0*'
_output_shapes
:?????????#2

Identity_1i

Identity_2Identityconcat_2:output:0*
T0*'
_output_shapes
:?????????#2

Identity_2i

Identity_3Identityconcat_3:output:0*
T0*'
_output_shapes
:?????????#2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*&
_input_shapes
:?????????#:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?I
?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_6424805

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
PartitionedCallX
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likee
mulMulinputsPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
muli
mul_1MulinputsPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
mul_1i
mul_2MulinputsPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
mul_2i
mul_3MulinputsPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	#?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3d
mul_4Mulstatesones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_4d
mul_5Mulstatesones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_5d
mul_6Mulstatesones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_6d
mul_7Mulstatesones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????#:??????????:??????????::::O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?6
s
'__inference__create_dropout_mask_134925

inputs
identity

identity_1

identity_2

identity_3{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
ellipsis_mask2
strided_sliceh
ones_like/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:????????? 2
	ones_like
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_1n
ones_like_1/ShapeShapestrided_slice_1:output:0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_2n
ones_like_2/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_2/Const?
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_3n
ones_like_3/ShapeShapestrided_slice_3:output:0*
T0*
_output_shapes
:2
ones_like_3/Shapek
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_3/Const?
ones_like_3Fillones_like_3/Shape:output:0ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_4n
ones_like_4/ShapeShapestrided_slice_4:output:0*
T0*
_output_shapes
:2
ones_like_4/Shapek
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_4/Const?
ones_like_4Fillones_like_4/Shape:output:0ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2ones_like:output:0ones_like_1:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/axis?
concat_1ConcatV2ones_like:output:0ones_like_2:output:0concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_1i
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV2ones_like:output:0ones_like_3:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_2i
concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_3/axis?
concat_3ConcatV2ones_like:output:0ones_like_4:output:0concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_3c
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????#2

Identityi

Identity_1Identityconcat_1:output:0*
T0*'
_output_shapes
:?????????#2

Identity_1i

Identity_2Identityconcat_2:output:0*
T0*'
_output_shapes
:?????????#2

Identity_2i

Identity_3Identityconcat_3:output:0*
T0*'
_output_shapes
:?????????#2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*&
_input_shapes
:?????????#:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?H
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6425152

inputs
lstm_cell_6425068
lstm_cell_6425070
lstm_cell_6425072
identity

identity_1

identity_2??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????#2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_6425068lstm_cell_6425070lstm_cell_6425072*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_64247032#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_6425068lstm_cell_6425070lstm_cell_6425072*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_6425081*
condR
while_cond_6425080*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identitywhile:output:5"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????#:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_6425535
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_6425535___redundant_placeholder0/
+while_cond_6425535___redundant_placeholder1/
+while_cond_6425535___redundant_placeholder2/
+while_cond_6425535___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?}
?
#__inference__traced_restore_6429269
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate4
0assignvariableop_5_dropout_lstm_lstm_cell_kernel>
:assignvariableop_6_dropout_lstm_lstm_cell_recurrent_kernel2
.assignvariableop_7_dropout_lstm_lstm_cell_bias2
.assignvariableop_8_mu_logstd_logmix_net_kernel0
,assignvariableop_9_mu_logstd_logmix_net_bias
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1
assignvariableop_14_total_2
assignvariableop_15_count_2
assignvariableop_16_total_3
assignvariableop_17_count_3<
8assignvariableop_18_adam_dropout_lstm_lstm_cell_kernel_mF
Bassignvariableop_19_adam_dropout_lstm_lstm_cell_recurrent_kernel_m:
6assignvariableop_20_adam_dropout_lstm_lstm_cell_bias_m:
6assignvariableop_21_adam_mu_logstd_logmix_net_kernel_m8
4assignvariableop_22_adam_mu_logstd_logmix_net_bias_m<
8assignvariableop_23_adam_dropout_lstm_lstm_cell_kernel_vF
Bassignvariableop_24_adam_dropout_lstm_lstm_cell_recurrent_kernel_v:
6assignvariableop_25_adam_dropout_lstm_lstm_cell_bias_v:
6assignvariableop_26_adam_mu_logstd_logmix_net_kernel_v8
4assignvariableop_27_adam_mu_logstd_logmix_net_bias_v
identity_29??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp0assignvariableop_5_dropout_lstm_lstm_cell_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp:assignvariableop_6_dropout_lstm_lstm_cell_recurrent_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_dropout_lstm_lstm_cell_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_mu_logstd_logmix_net_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_mu_logstd_logmix_net_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_2Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_2Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_3Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_3Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_dropout_lstm_lstm_cell_kernel_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpBassignvariableop_19_adam_dropout_lstm_lstm_cell_recurrent_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_dropout_lstm_lstm_cell_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_mu_logstd_logmix_net_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_mu_logstd_logmix_net_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adam_dropout_lstm_lstm_cell_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpBassignvariableop_24_adam_dropout_lstm_lstm_cell_recurrent_kernel_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_dropout_lstm_lstm_cell_bias_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_mu_logstd_logmix_net_kernel_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_mu_logstd_logmix_net_bias_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28?
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*?
_input_shapest
r: ::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
G__inference_sequential_layer_call_and_return_conditional_losses_6428777

inputs7
3mu_logstd_logmix_net_matmul_readvariableop_resource8
4mu_logstd_logmix_net_biasadd_readvariableop_resource
identity??
*mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp3mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*mu_logstd_logmix_net/MatMul/ReadVariableOp?
mu_logstd_logmix_net/MatMulMatMulinputs2mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mu_logstd_logmix_net/MatMul?
+mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp4mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
mu_logstd_logmix_net/BiasAddBiasAdd%mu_logstd_logmix_net/MatMul:product:03mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mu_logstd_logmix_net/BiasAddz
IdentityIdentity%mu_logstd_logmix_net/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
՝
?
while_body_6428307
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162#
!lstm_cell/StatefulPartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?س22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2Û?22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identitylstm_cell/mul_10:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_4?

Identity_5Identitylstm_cell/add_3:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
,__inference_sequential_layer_call_fn_6428796

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_64253552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6427866

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity

identity_1

identity_2??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????#2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162#
!lstm_cell/StatefulPartitionedCallt
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2è?22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_6427695*
condR
while_cond_6427694*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*-
_output_shapes
:???????????2

Identity?

Identity_1Identitywhile:output:4"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identitywhile:output:5"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????#:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
dropout_lstm_while_cond_6427345#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_6427345___redundant_placeholder0<
8dropout_lstm_while_cond_6427345___redundant_placeholder1<
8dropout_lstm_while_cond_6427345___redundant_placeholder2<
8dropout_lstm_while_cond_6427345___redundant_placeholder3
identity
e
LessLessplaceholder!less_dropout_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?%
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6426173

inputs
dropout_lstm_6426142
dropout_lstm_6426144
dropout_lstm_6426146
sequential_6426153
sequential_6426155
identity

identity_1

identity_2??$dropout_lstm/StatefulPartitionedCall?"sequential/StatefulPartitionedCall|
dropout_lstm/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:??????????#2
dropout_lstm/Cast?
$dropout_lstm/StatefulPartitionedCallStatefulPartitionedCalldropout_lstm/Cast:y:0dropout_lstm_6426142dropout_lstm_6426144dropout_lstm_6426146*
Tin
2*
Tout
2*U
_output_shapesC
A:???????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_64259662&
$dropout_lstm/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshape-dropout_lstm/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
"sequential/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0sequential_6426153sequential_6426155*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_64253732$
"sequential/StatefulPartitionedCall{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice+sequential/StatefulPartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice+sequential/StatefulPartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+sequential/StatefulPartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2?
IdentityIdentitystrided_slice:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_2:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitystrided_slice_1:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::2L
$dropout_lstm/StatefulPartitionedCall$dropout_lstm/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6426843

inputs8
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1

identity_2??dropout_lstm/while|
dropout_lstm/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:??????????#2
dropout_lstm/Castm
dropout_lstm/ShapeShapedropout_lstm/Cast:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape?
 dropout_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dropout_lstm/strided_slice/stack?
"dropout_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_1?
"dropout_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_2?
dropout_lstm/strided_sliceStridedSlicedropout_lstm/Shape:output:0)dropout_lstm/strided_slice/stack:output:0+dropout_lstm/strided_slice/stack_1:output:0+dropout_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slicew
dropout_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/mul/y?
dropout_lstm/zeros/mulMul#dropout_lstm/strided_slice:output:0!dropout_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/muly
dropout_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/Less/y?
dropout_lstm/zeros/LessLessdropout_lstm/zeros/mul:z:0"dropout_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/Less}
dropout_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/packed/1?
dropout_lstm/zeros/packedPack#dropout_lstm/strided_slice:output:0$dropout_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros/packedy
dropout_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros/Const?
dropout_lstm/zerosFill"dropout_lstm/zeros/packed:output:0!dropout_lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/mul/y?
dropout_lstm/zeros_1/mulMul#dropout_lstm/strided_slice:output:0#dropout_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/mul}
dropout_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/Less/y?
dropout_lstm/zeros_1/LessLessdropout_lstm/zeros_1/mul:z:0$dropout_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/Less?
dropout_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/packed/1?
dropout_lstm/zeros_1/packedPack#dropout_lstm/strided_slice:output:0&dropout_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros_1/packed}
dropout_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros_1/Const?
dropout_lstm/zeros_1Fill$dropout_lstm/zeros_1/packed:output:0#dropout_lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros_1?
dropout_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose/perm?
dropout_lstm/transpose	Transposedropout_lstm/Cast:y:0$dropout_lstm/transpose/perm:output:0*
T0*,
_output_shapes
:??????????#2
dropout_lstm/transposev
dropout_lstm/Shape_1Shapedropout_lstm/transpose:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape_1?
"dropout_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_1/stack?
$dropout_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_1?
$dropout_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_2?
dropout_lstm/strided_slice_1StridedSlicedropout_lstm/Shape_1:output:0+dropout_lstm/strided_slice_1/stack:output:0-dropout_lstm/strided_slice_1/stack_1:output:0-dropout_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slice_1?
(dropout_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(dropout_lstm/TensorArrayV2/element_shape?
dropout_lstm/TensorArrayV2TensorListReserve1dropout_lstm/TensorArrayV2/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2?
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   2D
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
4dropout_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordropout_lstm/transpose:y:0Kdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4dropout_lstm/TensorArrayUnstack/TensorListFromTensor?
"dropout_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_2/stack?
$dropout_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_1?
$dropout_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_2?
dropout_lstm/strided_slice_2StridedSlicedropout_lstm/transpose:y:0+dropout_lstm/strided_slice_2/stack:output:0-dropout_lstm/strided_slice_2/stack_1:output:0-dropout_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
&dropout_lstm/lstm_cell/PartitionedCallPartitionedCall%dropout_lstm/strided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442(
&dropout_lstm/lstm_cell/PartitionedCall?
&dropout_lstm/lstm_cell/ones_like/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/ones_like/Shape?
&dropout_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&dropout_lstm/lstm_cell/ones_like/Const?
 dropout_lstm/lstm_cell/ones_likeFill/dropout_lstm/lstm_cell/ones_like/Shape:output:0/dropout_lstm/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/ones_like?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_3~
dropout_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
dropout_lstm/lstm_cell/Const?
&dropout_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&dropout_lstm/lstm_cell/split/split_dim?
+dropout_lstm/lstm_cell/split/ReadVariableOpReadVariableOp4dropout_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_3?
dropout_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout_lstm/lstm_cell/Const_1?
(dropout_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(dropout_lstm/lstm_cell/split_1/split_dim?
-dropout_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp6dropout_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
*dropout_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*dropout_lstm/lstm_cell/strided_slice/stack?
,dropout_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$dropout_lstm/lstm_cell/strided_slice?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0-dropout_lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_3/stack?
.dropout_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_3/stack_1?
.dropout_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_3/stack_2?
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
*dropout_lstm/TensorArrayV2_1/element_shape?
dropout_lstm/TensorArrayV2_1TensorListReserve3dropout_lstm/TensorArrayV2_1/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2_1h
dropout_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_lstm/time?
%dropout_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%dropout_lstm/while/maximum_iterations?
dropout_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
dropout_lstm/while/loop_counter?
dropout_lstm/whileWhile(dropout_lstm/while/loop_counter:output:0.dropout_lstm/while/maximum_iterations:output:0dropout_lstm/time:output:0%dropout_lstm/TensorArrayV2_1:handle:0dropout_lstm/zeros:output:0dropout_lstm/zeros_1:output:0%dropout_lstm/strided_slice_1:output:0Ddropout_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:04dropout_lstm_lstm_cell_split_readvariableop_resource6dropout_lstm_lstm_cell_split_1_readvariableop_resource.dropout_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_6426684*+
cond#R!
dropout_lstm_while_cond_6426683*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype021
/dropout_lstm/TensorArrayV2Stack/TensorListStack?
"dropout_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"dropout_lstm/strided_slice_3/stack?
$dropout_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$dropout_lstm/strided_slice_3/stack_1?
$dropout_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_3/stack_2?
dropout_lstm/strided_slice_3StridedSlice8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+dropout_lstm/strided_slice_3/stack:output:0-dropout_lstm/strided_slice_3/stack_1:output:0-dropout_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
dropout_lstm/strided_slice_3?
dropout_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose_1/perm?
dropout_lstm/transpose_1	Transpose8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&dropout_lstm/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
dropout_lstm/transpose_1?
dropout_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/runtimeo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential/mu_logstd_logmix_net/BiasAdd{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2?
IdentityIdentitystrided_slice:output:0^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_2:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitystrided_slice_1:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::2(
dropout_lstm/whiledropout_lstm/while:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
dropout_lstm_while_cond_6427033#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_6427033___redundant_placeholder0<
8dropout_lstm_while_cond_6427033___redundant_placeholder1<
8dropout_lstm_while_cond_6427033___redundant_placeholder2<
8dropout_lstm_while_cond_6427033___redundant_placeholder3
identity
e
LessLessplaceholder!less_dropout_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
while_cond_6425218
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_6425218___redundant_placeholder0/
+while_cond_6425218___redundant_placeholder1/
+while_cond_6425218___redundant_placeholder2/
+while_cond_6425218___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6425966

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity

identity_1

identity_2??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????#2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
strided_slice_2?
lstm_cell/PartitionedCallPartitionedCallstrided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCallt
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_6425827*
condR
while_cond_6425826*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeq
IdentityIdentitytranspose_1:y:0^while*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????#:::2
whilewhile:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
֝
?
while_body_6425536
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162#
!lstm_cell/StatefulPartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identitylstm_cell/mul_10:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_4?

Identity_5Identitylstm_cell/add_3:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
??
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6426563

inputs8
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1

identity_2??.dropout_lstm/lstm_cell/StatefulPartitionedCall?dropout_lstm/while|
dropout_lstm/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:??????????#2
dropout_lstm/Castm
dropout_lstm/ShapeShapedropout_lstm/Cast:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape?
 dropout_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dropout_lstm/strided_slice/stack?
"dropout_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_1?
"dropout_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_2?
dropout_lstm/strided_sliceStridedSlicedropout_lstm/Shape:output:0)dropout_lstm/strided_slice/stack:output:0+dropout_lstm/strided_slice/stack_1:output:0+dropout_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slicew
dropout_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/mul/y?
dropout_lstm/zeros/mulMul#dropout_lstm/strided_slice:output:0!dropout_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/muly
dropout_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/Less/y?
dropout_lstm/zeros/LessLessdropout_lstm/zeros/mul:z:0"dropout_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/Less}
dropout_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/packed/1?
dropout_lstm/zeros/packedPack#dropout_lstm/strided_slice:output:0$dropout_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros/packedy
dropout_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros/Const?
dropout_lstm/zerosFill"dropout_lstm/zeros/packed:output:0!dropout_lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/mul/y?
dropout_lstm/zeros_1/mulMul#dropout_lstm/strided_slice:output:0#dropout_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/mul}
dropout_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/Less/y?
dropout_lstm/zeros_1/LessLessdropout_lstm/zeros_1/mul:z:0$dropout_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/Less?
dropout_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/packed/1?
dropout_lstm/zeros_1/packedPack#dropout_lstm/strided_slice:output:0&dropout_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros_1/packed}
dropout_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros_1/Const?
dropout_lstm/zeros_1Fill$dropout_lstm/zeros_1/packed:output:0#dropout_lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros_1?
dropout_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose/perm?
dropout_lstm/transpose	Transposedropout_lstm/Cast:y:0$dropout_lstm/transpose/perm:output:0*
T0*,
_output_shapes
:??????????#2
dropout_lstm/transposev
dropout_lstm/Shape_1Shapedropout_lstm/transpose:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape_1?
"dropout_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_1/stack?
$dropout_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_1?
$dropout_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_2?
dropout_lstm/strided_slice_1StridedSlicedropout_lstm/Shape_1:output:0+dropout_lstm/strided_slice_1/stack:output:0-dropout_lstm/strided_slice_1/stack_1:output:0-dropout_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slice_1?
(dropout_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(dropout_lstm/TensorArrayV2/element_shape?
dropout_lstm/TensorArrayV2TensorListReserve1dropout_lstm/TensorArrayV2/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2?
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   2D
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
4dropout_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordropout_lstm/transpose:y:0Kdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4dropout_lstm/TensorArrayUnstack/TensorListFromTensor?
"dropout_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_2/stack?
$dropout_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_1?
$dropout_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_2?
dropout_lstm/strided_slice_2StridedSlicedropout_lstm/transpose:y:0+dropout_lstm/strided_slice_2/stack:output:0-dropout_lstm/strided_slice_2/stack_1:output:0-dropout_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
.dropout_lstm/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall%dropout_lstm/strided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_13061620
.dropout_lstm/lstm_cell/StatefulPartitionedCall?
&dropout_lstm/lstm_cell/ones_like/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/ones_like/Shape?
&dropout_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&dropout_lstm/lstm_cell/ones_like/Const?
 dropout_lstm/lstm_cell/ones_likeFill/dropout_lstm/lstm_cell/ones_like/Shape:output:0/dropout_lstm/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/ones_like?
$dropout_lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2&
$dropout_lstm/lstm_cell/dropout/Const?
"dropout_lstm/lstm_cell/dropout/MulMul)dropout_lstm/lstm_cell/ones_like:output:0-dropout_lstm/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2$
"dropout_lstm/lstm_cell/dropout/Mul?
$dropout_lstm/lstm_cell/dropout/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$dropout_lstm/lstm_cell/dropout/Shape?
;dropout_lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform-dropout_lstm/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2=
;dropout_lstm/lstm_cell/dropout/random_uniform/RandomUniform?
-dropout_lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2/
-dropout_lstm/lstm_cell/dropout/GreaterEqual/y?
+dropout_lstm/lstm_cell/dropout/GreaterEqualGreaterEqualDdropout_lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:06dropout_lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2-
+dropout_lstm/lstm_cell/dropout/GreaterEqual?
#dropout_lstm/lstm_cell/dropout/CastCast/dropout_lstm/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2%
#dropout_lstm/lstm_cell/dropout/Cast?
$dropout_lstm/lstm_cell/dropout/Mul_1Mul&dropout_lstm/lstm_cell/dropout/Mul:z:0'dropout_lstm/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout/Mul_1?
&dropout_lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_1/Const?
$dropout_lstm/lstm_cell/dropout_1/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_1/Mul?
&dropout_lstm/lstm_cell/dropout_1/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_1/Shape?
=dropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_1/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_1/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_1/CastCast1dropout_lstm/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_1/Cast?
&dropout_lstm/lstm_cell/dropout_1/Mul_1Mul(dropout_lstm/lstm_cell/dropout_1/Mul:z:0)dropout_lstm/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_1/Mul_1?
&dropout_lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_2/Const?
$dropout_lstm/lstm_cell/dropout_2/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_2/Mul?
&dropout_lstm/lstm_cell/dropout_2/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_2/Shape?
=dropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_2/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_2/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_2/CastCast1dropout_lstm/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_2/Cast?
&dropout_lstm/lstm_cell/dropout_2/Mul_1Mul(dropout_lstm/lstm_cell/dropout_2/Mul:z:0)dropout_lstm/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_2/Mul_1?
&dropout_lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_3/Const?
$dropout_lstm/lstm_cell/dropout_3/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_3/Mul?
&dropout_lstm/lstm_cell/dropout_3/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_3/Shape?
=dropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ə2?
=dropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_3/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_3/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_3/CastCast1dropout_lstm/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_3/Cast?
&dropout_lstm/lstm_cell/dropout_3/Mul_1Mul(dropout_lstm/lstm_cell/dropout_3/Mul:z:0)dropout_lstm/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_3/Mul_1?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:07dropout_lstm/lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:07dropout_lstm/lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:07dropout_lstm/lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:07dropout_lstm/lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_3~
dropout_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
dropout_lstm/lstm_cell/Const?
&dropout_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&dropout_lstm/lstm_cell/split/split_dim?
+dropout_lstm/lstm_cell/split/ReadVariableOpReadVariableOp4dropout_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_3?
dropout_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout_lstm/lstm_cell/Const_1?
(dropout_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(dropout_lstm/lstm_cell/split_1/split_dim?
-dropout_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp6dropout_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0(dropout_lstm/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
*dropout_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*dropout_lstm/lstm_cell/strided_slice/stack?
,dropout_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$dropout_lstm/lstm_cell/strided_slice?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0-dropout_lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_3/stack?
.dropout_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_3/stack_1?
.dropout_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_3/stack_2?
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
*dropout_lstm/TensorArrayV2_1/element_shape?
dropout_lstm/TensorArrayV2_1TensorListReserve3dropout_lstm/TensorArrayV2_1/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2_1h
dropout_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_lstm/time?
%dropout_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%dropout_lstm/while/maximum_iterations?
dropout_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
dropout_lstm/while/loop_counter?
dropout_lstm/whileWhile(dropout_lstm/while/loop_counter:output:0.dropout_lstm/while/maximum_iterations:output:0dropout_lstm/time:output:0%dropout_lstm/TensorArrayV2_1:handle:0dropout_lstm/zeros:output:0dropout_lstm/zeros_1:output:0%dropout_lstm/strided_slice_1:output:0Ddropout_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:04dropout_lstm_lstm_cell_split_readvariableop_resource6dropout_lstm_lstm_cell_split_1_readvariableop_resource.dropout_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_6426372*+
cond#R!
dropout_lstm_while_cond_6426371*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype021
/dropout_lstm/TensorArrayV2Stack/TensorListStack?
"dropout_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"dropout_lstm/strided_slice_3/stack?
$dropout_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$dropout_lstm/strided_slice_3/stack_1?
$dropout_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_3/stack_2?
dropout_lstm/strided_slice_3StridedSlice8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+dropout_lstm/strided_slice_3/stack:output:0-dropout_lstm/strided_slice_3/stack_1:output:0-dropout_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
dropout_lstm/strided_slice_3?
dropout_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose_1/perm?
dropout_lstm/transpose_1	Transpose8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&dropout_lstm/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
dropout_lstm/transpose_1?
dropout_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/runtimeo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential/mu_logstd_logmix_net/BiasAdd{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2?
IdentityIdentitystrided_slice:output:0/^dropout_lstm/lstm_cell/StatefulPartitionedCall^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_2:output:0/^dropout_lstm/lstm_cell/StatefulPartitionedCall^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitystrided_slice_1:output:0/^dropout_lstm/lstm_cell/StatefulPartitionedCall^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::2`
.dropout_lstm/lstm_cell/StatefulPartitionedCall.dropout_lstm/lstm_cell/StatefulPartitionedCall2(
dropout_lstm/whiledropout_lstm/while:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
՝
?
while_body_6427695
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162#
!lstm_cell/StatefulPartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??s22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?Đ22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identitylstm_cell/mul_10:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_4?

Identity_5Identitylstm_cell/add_3:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
,__inference_sequential_layer_call_fn_6428805

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_64253732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_mdnrnn_layer_call_fn_6427543
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*N
_output_shapes<
::??????????:?????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_mdnrnn_layer_call_and_return_conditional_losses_64261732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????#
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6427505
input_18
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1

identity_2??dropout_lstm/while}
dropout_lstm/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:??????????#2
dropout_lstm/Castm
dropout_lstm/ShapeShapedropout_lstm/Cast:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape?
 dropout_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dropout_lstm/strided_slice/stack?
"dropout_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_1?
"dropout_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dropout_lstm/strided_slice/stack_2?
dropout_lstm/strided_sliceStridedSlicedropout_lstm/Shape:output:0)dropout_lstm/strided_slice/stack:output:0+dropout_lstm/strided_slice/stack_1:output:0+dropout_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slicew
dropout_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/mul/y?
dropout_lstm/zeros/mulMul#dropout_lstm/strided_slice:output:0!dropout_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/muly
dropout_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/Less/y?
dropout_lstm/zeros/LessLessdropout_lstm/zeros/mul:z:0"dropout_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros/Less}
dropout_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros/packed/1?
dropout_lstm/zeros/packedPack#dropout_lstm/strided_slice:output:0$dropout_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros/packedy
dropout_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros/Const?
dropout_lstm/zerosFill"dropout_lstm/zeros/packed:output:0!dropout_lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/mul/y?
dropout_lstm/zeros_1/mulMul#dropout_lstm/strided_slice:output:0#dropout_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/mul}
dropout_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/Less/y?
dropout_lstm/zeros_1/LessLessdropout_lstm/zeros_1/mul:z:0$dropout_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
dropout_lstm/zeros_1/Less?
dropout_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
dropout_lstm/zeros_1/packed/1?
dropout_lstm/zeros_1/packedPack#dropout_lstm/strided_slice:output:0&dropout_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
dropout_lstm/zeros_1/packed}
dropout_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/zeros_1/Const?
dropout_lstm/zeros_1Fill$dropout_lstm/zeros_1/packed:output:0#dropout_lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/zeros_1?
dropout_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose/perm?
dropout_lstm/transpose	Transposedropout_lstm/Cast:y:0$dropout_lstm/transpose/perm:output:0*
T0*,
_output_shapes
:??????????#2
dropout_lstm/transposev
dropout_lstm/Shape_1Shapedropout_lstm/transpose:y:0*
T0*
_output_shapes
:2
dropout_lstm/Shape_1?
"dropout_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_1/stack?
$dropout_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_1?
$dropout_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_1/stack_2?
dropout_lstm/strided_slice_1StridedSlicedropout_lstm/Shape_1:output:0+dropout_lstm/strided_slice_1/stack:output:0-dropout_lstm/strided_slice_1/stack_1:output:0-dropout_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout_lstm/strided_slice_1?
(dropout_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(dropout_lstm/TensorArrayV2/element_shape?
dropout_lstm/TensorArrayV2TensorListReserve1dropout_lstm/TensorArrayV2/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2?
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   2D
Bdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
4dropout_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordropout_lstm/transpose:y:0Kdropout_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4dropout_lstm/TensorArrayUnstack/TensorListFromTensor?
"dropout_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dropout_lstm/strided_slice_2/stack?
$dropout_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_1?
$dropout_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_2/stack_2?
dropout_lstm/strided_slice_2StridedSlicedropout_lstm/transpose:y:0+dropout_lstm/strided_slice_2/stack:output:0-dropout_lstm/strided_slice_2/stack_1:output:0-dropout_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
&dropout_lstm/lstm_cell/PartitionedCallPartitionedCall%dropout_lstm/strided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442(
&dropout_lstm/lstm_cell/PartitionedCall?
&dropout_lstm/lstm_cell/ones_like/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/ones_like/Shape?
&dropout_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&dropout_lstm/lstm_cell/ones_like/Const?
 dropout_lstm/lstm_cell/ones_likeFill/dropout_lstm/lstm_cell/ones_like/Shape:output:0/dropout_lstm/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/ones_like?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:0/dropout_lstm/lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
dropout_lstm/lstm_cell/mul_3~
dropout_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
dropout_lstm/lstm_cell/Const?
&dropout_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&dropout_lstm/lstm_cell/split/split_dim?
+dropout_lstm/lstm_cell/split/ReadVariableOpReadVariableOp4dropout_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_3?
dropout_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout_lstm/lstm_cell/Const_1?
(dropout_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(dropout_lstm/lstm_cell/split_1/split_dim?
-dropout_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp6dropout_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0)dropout_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
*dropout_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*dropout_lstm/lstm_cell/strided_slice/stack?
,dropout_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$dropout_lstm/lstm_cell/strided_slice?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0-dropout_lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_3/stack?
.dropout_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_3/stack_1?
.dropout_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_3/stack_2?
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
*dropout_lstm/TensorArrayV2_1/element_shape?
dropout_lstm/TensorArrayV2_1TensorListReserve3dropout_lstm/TensorArrayV2_1/element_shape:output:0%dropout_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dropout_lstm/TensorArrayV2_1h
dropout_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_lstm/time?
%dropout_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%dropout_lstm/while/maximum_iterations?
dropout_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
dropout_lstm/while/loop_counter?
dropout_lstm/whileWhile(dropout_lstm/while/loop_counter:output:0.dropout_lstm/while/maximum_iterations:output:0dropout_lstm/time:output:0%dropout_lstm/TensorArrayV2_1:handle:0dropout_lstm/zeros:output:0dropout_lstm/zeros_1:output:0%dropout_lstm/strided_slice_1:output:0Ddropout_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:04dropout_lstm_lstm_cell_split_readvariableop_resource6dropout_lstm_lstm_cell_split_1_readvariableop_resource.dropout_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_6427346*+
cond#R!
dropout_lstm_while_cond_6427345*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype021
/dropout_lstm/TensorArrayV2Stack/TensorListStack?
"dropout_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"dropout_lstm/strided_slice_3/stack?
$dropout_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$dropout_lstm/strided_slice_3/stack_1?
$dropout_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$dropout_lstm/strided_slice_3/stack_2?
dropout_lstm/strided_slice_3StridedSlice8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+dropout_lstm/strided_slice_3/stack:output:0-dropout_lstm/strided_slice_3/stack_1:output:0-dropout_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
dropout_lstm/strided_slice_3?
dropout_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dropout_lstm/transpose_1/perm?
dropout_lstm/transpose_1	Transpose8dropout_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&dropout_lstm/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
dropout_lstm/transpose_1?
dropout_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout_lstm/runtimeo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential/mu_logstd_logmix_net/BiasAdd{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2?
IdentityIdentitystrided_slice:output:0^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_2:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitystrided_slice_1:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::2(
dropout_lstm/whiledropout_lstm/while:U Q
,
_output_shapes
:??????????#
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_mdnrnn_layer_call_fn_6427524
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*N
_output_shapes<
::??????????:?????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_mdnrnn_layer_call_and_return_conditional_losses_64261192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????#
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
? 
?
while_body_6425081
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_6425105_0
lstm_cell_6425107_0
lstm_cell_6425109_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_6425105
lstm_cell_6425107
lstm_cell_6425109??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_6425105_0lstm_cell_6425107_0lstm_cell_6425109_0*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_64247032#
!lstm_cell/StatefulPartitionedCall?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_4?

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"(
lstm_cell_6425105lstm_cell_6425105_0"(
lstm_cell_6425107lstm_cell_6425107_0"(
lstm_cell_6425109lstm_cell_6425109_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?o
?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_6424703

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162
StatefulPartitionedCallX
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const?
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ǂ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/Mul_1m
mulMulinputs StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
mulq
mul_1Mulinputs StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
mul_1q
mul_2Mulinputs StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
mul_2q
mul_3Mulinputs StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	#?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3c
mul_4Mulstatesdropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4e
mul_5Mulstatesdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5e
mul_6Mulstatesdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6e
mul_7Mulstatesdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10y
IdentityIdentity
mul_10:z:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity}

Identity_1Identity
mul_10:z:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1|

Identity_2Identity	add_3:z:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????#:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
G__inference_sequential_layer_call_and_return_conditional_losses_6425373

inputs 
mu_logstd_logmix_net_6425367 
mu_logstd_logmix_net_6425369
identity??,mu_logstd_logmix_net/StatefulPartitionedCall?
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinputsmu_logstd_logmix_net_6425367mu_logstd_logmix_net_6425369*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_64253172.
,mu_logstd_logmix_net/StatefulPartitionedCall?
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0-^mu_logstd_logmix_net/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?H
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6425290

inputs
lstm_cell_6425206
lstm_cell_6425208
lstm_cell_6425210
identity

identity_1

identity_2??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????#2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_6425206lstm_cell_6425208lstm_cell_6425210*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_64248052#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_6425206lstm_cell_6425208lstm_cell_6425210*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_6425219*
condR
while_cond_6425218*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identitywhile:output:5"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????#:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_mdnrnn_layer_call_fn_6426881

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*N
_output_shapes<
::??????????:?????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_mdnrnn_layer_call_and_return_conditional_losses_64261732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428478
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity

identity_1

identity_2??!lstm_cell/StatefulPartitionedCall?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????#2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162#
!lstm_cell/StatefulPartitionedCallt
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ӝ20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2႕22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ڎ!22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ۆ22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0*lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_6428307*
condR
while_cond_6428306*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identitywhile:output:5"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????#:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????#
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
G__inference_sequential_layer_call_and_return_conditional_losses_6425355

inputs 
mu_logstd_logmix_net_6425349 
mu_logstd_logmix_net_6425351
identity??,mu_logstd_logmix_net/StatefulPartitionedCall?
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinputsmu_logstd_logmix_net_6425349mu_logstd_logmix_net_6425351*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_64253172.
,mu_logstd_logmix_net/StatefulPartitionedCall?
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0-^mu_logstd_logmix_net/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?	
?
dropout_lstm_while_cond_6426371#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_6426371___redundant_placeholder0<
8dropout_lstm_while_cond_6426371___redundant_placeholder1<
8dropout_lstm_while_cond_6426371___redundant_placeholder2<
8dropout_lstm_while_cond_6426371___redundant_placeholder3
identity
e
LessLessplaceholder!less_dropout_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
? 
?
while_body_6425219
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_6425243_0
lstm_cell_6425245_0
lstm_cell_6425247_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_6425243
lstm_cell_6425245
lstm_cell_6425247??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_6425243_0lstm_cell_6425245_0lstm_cell_6425247_0*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_64248052#
!lstm_cell/StatefulPartitionedCall?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_4?

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"(
lstm_cell_6425243lstm_cell_6425243_0"(
lstm_cell_6425245lstm_cell_6425245_0"(
lstm_cell_6425247lstm_cell_6425247_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?p
?
dropout_lstm_while_body_6427346#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
dropout_lstm_strided_slice_1_0^
Ztensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5 
dropout_lstm_strided_slice_1\
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/PartitionedCallPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yk
add_1AddV2dropout_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityl

Identity_1Identity%dropout_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_5">
dropout_lstm_strided_slice_1dropout_lstm_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"?
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensorZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
while_cond_6425826
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_6425826___redundant_placeholder0/
+while_cond_6425826___redundant_placeholder1/
+while_cond_6425826___redundant_placeholder2/
+while_cond_6425826___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
while_cond_6428306
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_6428306___redundant_placeholder0/
+while_cond_6428306___redundant_placeholder1/
+while_cond_6428306___redundant_placeholder2/
+while_cond_6428306___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?o
?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_6428956

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162
StatefulPartitionedCallZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const?
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??_2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??q2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??a2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/Mul_1m
mulMulinputs StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
mulq
mul_1Mulinputs StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
mul_1q
mul_2Mulinputs StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
mul_2q
mul_3Mulinputs StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	#?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3e
mul_4Mulstates_0dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4g
mul_5Mulstates_0dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5g
mul_6Mulstates_0dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6g
mul_7Mulstates_0dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10y
IdentityIdentity
mul_10:z:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity}

Identity_1Identity
mul_10:z:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1|

Identity_2Identity	add_3:z:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????#:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?n
?
while_body_6427986
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/PartitionedCallPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
(__inference_mdnrnn_layer_call_fn_6426862

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*N
_output_shapes<
::??????????:?????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_mdnrnn_layer_call_and_return_conditional_losses_64261192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_layer_call_fn_6425362
input_1
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_64253552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
?	
?
G__inference_sequential_layer_call_and_return_conditional_losses_6425334
input_1 
mu_logstd_logmix_net_6425328 
mu_logstd_logmix_net_6425330
identity??,mu_logstd_logmix_net/StatefulPartitionedCall?
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinput_1mu_logstd_logmix_net_6425328mu_logstd_logmix_net_6425330*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_64253172.
,mu_logstd_logmix_net/StatefulPartitionedCall?
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0-^mu_logstd_logmix_net/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
?W
v
'__inference__create_dropout_mask_135007

inputs
identity

identity_1

identity_2

identity_3?{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:d *

begin_mask*
ellipsis_mask2
strided_slices
ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d       2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const{
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*
_output_shapes

:d 2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Constv
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*
_output_shapes

:d 2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d       2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes

:d *
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:d 2
dropout/GreaterEqualv
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:d 2
dropout/Castq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes

:d 2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/Const|
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*
_output_shapes

:d 2
dropout_1/Muls
dropout_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d       2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*
_output_shapes

:d *
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*
_output_shapes

:d 2
dropout_1/GreaterEqual|
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:d 2
dropout_1/Casty
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*
_output_shapes

:d 2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/Const|
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*
_output_shapes

:d 2
dropout_2/Muls
dropout_2/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d       2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*
_output_shapes

:d *
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*
_output_shapes

:d 2
dropout_2/GreaterEqual|
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:d 2
dropout_2/Casty
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*
_output_shapes

:d 2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_3/Const|
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*
_output_shapes

:d 2
dropout_3/Muls
dropout_3/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d       2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*
_output_shapes

:d *
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*
_output_shapes

:d 2
dropout_3/GreaterEqual|
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:d 2
dropout_3/Casty
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*
_output_shapes

:d 2
dropout_3/Mul_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
ellipsis_mask*
end_mask2
strided_slice_1w
ones_like_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d      2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:d2
ones_like_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
ellipsis_mask*
end_mask2
strided_slice_2w
ones_like_2/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d      2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_2/Const?
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:d2
ones_like_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
ellipsis_mask*
end_mask2
strided_slice_3w
ones_like_3/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d      2
ones_like_3/Shapek
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_3/Const?
ones_like_3Fillones_like_3/Shape:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:d2
ones_like_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
ellipsis_mask*
end_mask2
strided_slice_4w
ones_like_4/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d      2
ones_like_4/Shapek
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_4/Const?
ones_like_4Fillones_like_4/Shape:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:d2
ones_like_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2dropout/Mul_1:z:0ones_like_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:d#2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/axis?
concat_1ConcatV2dropout_1/Mul_1:z:0ones_like_2:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:d#2

concat_1i
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV2dropout_2/Mul_1:z:0ones_like_3:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes

:d#2

concat_2i
concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_3/axis?
concat_3ConcatV2dropout_3/Mul_1:z:0ones_like_4:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes

:d#2

concat_3Z
IdentityIdentityconcat:output:0*
T0*
_output_shapes

:d#2

Identity`

Identity_1Identityconcat_1:output:0*
T0*
_output_shapes

:d#2

Identity_1`

Identity_2Identityconcat_2:output:0*
T0*
_output_shapes

:d#2

Identity_2`

Identity_3Identityconcat_3:output:0*
T0*
_output_shapes

:d#2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes

:d#:F B

_output_shapes

:d#
 
_user_specified_nameinputs
֖
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428737
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity

identity_1

identity_2??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????#2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????#*
shrink_axis_mask2
strided_slice_2?
lstm_cell/PartitionedCallPartitionedCallstrided_slice_2:output:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCallt
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_6428598*
condR
while_cond_6428597*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimey
IdentityIdentitytranspose_1:y:0^while*
T0*5
_output_shapes#
!:???????????????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????#:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????#
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
dropout_lstm_while_cond_6424381#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_6424381___redundant_placeholder0<
8dropout_lstm_while_cond_6424381___redundant_placeholder1<
8dropout_lstm_while_cond_6424381___redundant_placeholder2<
8dropout_lstm_while_cond_6424381___redundant_placeholder3
identity
e
LessLessplaceholder!less_dropout_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?Y
v
'__inference__create_dropout_mask_130616

inputs
identity

identity_1

identity_2

identity_3?{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
ellipsis_mask2
strided_sliceh
ones_like/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:????????? 2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2Ģ?2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_3/Mul_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_1n
ones_like_1/ShapeShapestrided_slice_1:output:0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_2n
ones_like_2/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_2/Const?
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_3n
ones_like_3/ShapeShapestrided_slice_3:output:0*
T0*
_output_shapes
:2
ones_like_3/Shapek
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_3/Const?
ones_like_3Fillones_like_3/Shape:output:0ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_4n
ones_like_4/ShapeShapestrided_slice_4:output:0*
T0*
_output_shapes
:2
ones_like_4/Shapek
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_4/Const?
ones_like_4Fillones_like_4/Shape:output:0ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2dropout/Mul_1:z:0ones_like_1:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/axis?
concat_1ConcatV2dropout_1/Mul_1:z:0ones_like_2:output:0concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_1i
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV2dropout_2/Mul_1:z:0ones_like_3:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_2i
concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_3/axis?
concat_3ConcatV2dropout_3/Mul_1:z:0ones_like_4:output:0concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_3c
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????#2

Identityi

Identity_1Identityconcat_1:output:0*
T0*'
_output_shapes
:?????????#2

Identity_1i

Identity_2Identityconcat_2:output:0*
T0*'
_output_shapes
:?????????#2

Identity_2i

Identity_3Identityconcat_3:output:0*
T0*'
_output_shapes
:?????????#2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*&
_input_shapes
:?????????#:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
.__inference_dropout_lstm_layer_call_fn_6428140

inputs
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*U
_output_shapesC
A:???????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_64257072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????#:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_6427985
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_6427985___redundant_placeholder0/
+while_cond_6427985___redundant_placeholder1/
+while_cond_6427985___redundant_placeholder2/
+while_cond_6427985___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
,__inference_sequential_layer_call_fn_6425380
input_1
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_64253732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
??
?
dropout_lstm_while_body_6427034#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
dropout_lstm_strided_slice_1_0^
Ztensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5 
dropout_lstm_strided_slice_1\
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1306162#
!lstm_cell/StatefulPartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?Ӡ22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ئ=22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_3/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0*lstm_cell/StatefulPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yk
add_1AddV2dropout_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity?

Identity_1Identity%dropout_lstm_while_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identitylstm_cell/mul_10:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_4?

Identity_5Identitylstm_cell/add_3:z:0"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5">
dropout_lstm_strided_slice_1dropout_lstm_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"?
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensorZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?6
s
'__inference__create_dropout_mask_130244

inputs
identity

identity_1

identity_2

identity_3{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *

begin_mask*
ellipsis_mask2
strided_sliceh
ones_like/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:????????? 2
	ones_like
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_1n
ones_like_1/ShapeShapestrided_slice_1:output:0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_2n
ones_like_2/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
ones_like_2/Shapek
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_2/Const?
ones_like_2Fillones_like_2/Shape:output:0ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_3n
ones_like_3/ShapeShapestrided_slice_3:output:0*
T0*
_output_shapes
:2
ones_like_3/Shapek
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_3/Const?
ones_like_3Fillones_like_3/Shape:output:0ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
strided_slice_4n
ones_like_4/ShapeShapestrided_slice_4:output:0*
T0*
_output_shapes
:2
ones_like_4/Shapek
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_4/Const?
ones_like_4Fillones_like_4/Shape:output:0ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
ones_like_4e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2ones_like:output:0ones_like_1:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/axis?
concat_1ConcatV2ones_like:output:0ones_like_2:output:0concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_1i
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV2ones_like:output:0ones_like_3:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_2i
concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_3/axis?
concat_3ConcatV2ones_like:output:0ones_like_4:output:0concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2

concat_3c
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????#2

Identityi

Identity_1Identityconcat_1:output:0*
T0*'
_output_shapes
:?????????#2

Identity_1i

Identity_2Identityconcat_2:output:0*
T0*'
_output_shapes
:?????????#2

Identity_2i

Identity_3Identityconcat_3:output:0*
T0*'
_output_shapes
:?????????#2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*&
_input_shapes
:?????????#:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?I
?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_6429041

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
PartitionedCallZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likee
mulMulinputsPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
muli
mul_1MulinputsPartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
mul_1i
mul_2MulinputsPartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
mul_2i
mul_3MulinputsPartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	#?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3f
mul_4Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_4f
mul_5Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_5f
mul_6Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_6f
mul_7Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????#:??????????:??????????::::O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_lstm_cell_layer_call_fn_6428839

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_64248052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????#:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?%
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6426119

inputs
dropout_lstm_6426088
dropout_lstm_6426090
dropout_lstm_6426092
sequential_6426099
sequential_6426101
identity

identity_1

identity_2??$dropout_lstm/StatefulPartitionedCall?"sequential/StatefulPartitionedCall|
dropout_lstm/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:??????????#2
dropout_lstm/Cast?
$dropout_lstm/StatefulPartitionedCallStatefulPartitionedCalldropout_lstm/Cast:y:0dropout_lstm_6426088dropout_lstm_6426090dropout_lstm_6426092*
Tin
2*
Tout
2*U
_output_shapesC
A:???????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_64257072&
$dropout_lstm/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshape-dropout_lstm/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
"sequential/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0sequential_6426099sequential_6426101*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_64253552$
"sequential/StatefulPartitionedCall{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice+sequential/StatefulPartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice+sequential/StatefulPartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+sequential/StatefulPartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_2?
IdentityIdentitystrided_slice:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_2:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitystrided_slice_1:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????#:::::2L
$dropout_lstm/StatefulPartitionedCall$dropout_lstm/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
6__inference_mu_logstd_logmix_net_layer_call_fn_6429060

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_64253172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?p
?
dropout_lstm_while_body_6424382#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
dropout_lstm_strided_slice_1_0^
Ztensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5 
dropout_lstm_strided_slice_1\
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource??
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????#   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????#*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
lstm_cell/PartitionedCallPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*`
_output_shapesN
L:?????????#:?????????#:?????????#:?????????#* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*0
f+R)
'__inference__create_dropout_mask_1302442
lstm_cell/PartitionedCalls
lstm_cell/ones_like/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like/Const?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:1*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:2*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0"lstm_cell/PartitionedCall:output:3*
T0*'
_output_shapes
:?????????#2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	#?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	#?:	#?:	#?:	#?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yk
add_1AddV2dropout_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityl

Identity_1Identity%dropout_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3m

Identity_4Identitylstm_cell/mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_5">
dropout_lstm_strided_slice_1dropout_lstm_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"?
Xtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensorZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
?
?
while_cond_6428597
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_6428597___redundant_placeholder0/
+while_cond_6428597___redundant_placeholder1/
+while_cond_6428597___redundant_placeholder2/
+while_cond_6428597___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
.__inference_dropout_lstm_layer_call_fn_6428155

inputs
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*U
_output_shapesC
A:???????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_64259662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????#:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????#8
MDN1
StatefulPartitionedCall:0??????????5
d0
StatefulPartitionedCall:1?????????5
r0
StatefulPartitionedCall:2?????????tensorflow/serving/predict:??
?
	optimizer
loss_fn
inference_base
out_net
loss
trainable_variables

signatures
	keras_api
		variables

regularization_losses
*`&call_and_return_all_conditional_losses
a__call__
b_default_save_signature"?
_tf_keras_model?{"trainable": true, "model_config": {"class_name": "MDNRNN"}, "is_graph_network": false, "dtype": "float32", "backend": "tensorflow", "name": "mdnrnn", "expects_training_arg": true, "training_config": {"loss": {"r": "r_loss_func", "d": "d_loss_func", "MDN": "z_loss_func"}, "optimizer_config": {"class_name": "Adam", "config": {"epsilon": 1e-07, "beta_1": 0.8999999761581421, "clipvalue": 1.0, "learning_rate": 0.0010000000474974513, "amsgrad": false, "decay": 0.0, "name": "Adam", "beta_2": 0.9990000128746033}}, "sample_weight_mode": null, "weighted_metrics": null, "loss_weights": null, "metrics": null}, "config": {"layer was saved without config": true}, "class_name": "MDNRNN", "keras_version": "2.3.0-tf", "batch_input_shape": null}
?
iter

beta_1

beta_2
	decay
learning_ratemVmWmXmYmZv[v\v]v^v_"
	optimizer
 "
trackable_dict_wrapper
?
cell

state_spec
trainable_variables
	keras_api
	variables
regularization_losses
*c&call_and_return_all_conditional_losses
d__call__
e_create_dropout_mask"?

_tf_keras_rnn_layer?	{"trainable": true, "dtype": "float32", "batch_input_shape": null, "expects_training_arg": true, "class_name": "DropoutLSTM", "stateful": false, "build_input_shape": {"items": [100, 1000, 35], "class_name": "TensorShape"}, "input_spec": [{"class_name": "InputSpec", "config": {"max_ndim": null, "axes": {}, "min_ndim": null, "dtype": null, "shape": {"items": [null, null, 35], "class_name": "__tuple__"}, "ndim": 3}}], "name": "dropout_lstm", "config": {"trainable": true, "activation": "tanh", "unit_forget_bias": true, "kernel_constraint": null, "activity_regularizer": null, "recurrent_activation": "sigmoid", "unroll": false, "dropout": 0.05, "bias_initializer": {"class_name": "Zeros", "config": {}}, "recurrent_dropout": 0.05, "go_backwards": false, "recurrent_constraint": null, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_constraint": null, "recurrent_regularizer": null, "units": 256, "use_bias": true, "dtype": "float32", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "return_state": true, "name": "dropout_lstm", "implementation": 1, "return_sequences": true, "kernel_regularizer": null, "stateful": false, "bias_regularizer": null, "time_major": false}}
?
layer_with_weights-0
layer-0
trainable_variables
	keras_api
	variables
regularization_losses
*f&call_and_return_all_conditional_losses
g__call__"?
_tf_keras_sequential?{"trainable": true, "build_input_shape": {"items": [null, 256], "class_name": "TensorShape"}, "model_config": {"class_name": "Sequential", "config": {"build_input_shape": {"items": [null, 256], "class_name": "TensorShape"}, "name": "sequential", "layers": [{"class_name": "Dense", "config": {"trainable": true, "units": 482, "kernel_constraint": null, "activity_regularizer": null, "activation": "linear", "dtype": "float32", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "use_bias": true, "name": "mu_logstd_logmix_net", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "batch_input_shape": {"items": [null, 256], "class_name": "__tuple__"}, "bias_constraint": null, "bias_regularizer": null}}]}}, "is_graph_network": true, "dtype": "float32", "class_name": "Sequential", "name": "sequential", "config": {"build_input_shape": {"items": [null, 256], "class_name": "TensorShape"}, "name": "sequential", "layers": [{"class_name": "Dense", "config": {"trainable": true, "units": 482, "kernel_constraint": null, "activity_regularizer": null, "activation": "linear", "dtype": "float32", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "use_bias": true, "name": "mu_logstd_logmix_net", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "batch_input_shape": {"items": [null, 256], "class_name": "__tuple__"}, "bias_constraint": null, "bias_regularizer": null}}]}, "batch_input_shape": null, "backend": "tensorflow", "input_spec": {"class_name": "InputSpec", "config": {"max_ndim": null, "axes": {"-1": 256}, "min_ndim": 2, "dtype": null, "shape": null, "ndim": null}}, "expects_training_arg": true, "keras_version": "2.3.0-tf"}
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
,
hserving_default"
signature_map
?
 layer_metrics
trainable_variables
		variables
!non_trainable_variables
"metrics
#layer_regularization_losses

regularization_losses

$layers
&`"call_and_return_conditional_losses
*`&call_and_return_all_conditional_losses
a__call__
b_default_save_signature"
_generic_user_object
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?

kernel
recurrent_kernel
bias
%trainable_variables
&	keras_api
'	variables
(regularization_losses
*i&call_and_return_all_conditional_losses
j__call__"?
_tf_keras_layer?{"trainable": true, "dtype": "float32", "batch_input_shape": null, "expects_training_arg": true, "stateful": false, "class_name": "LSTMCell", "name": "lstm_cell", "config": {"trainable": true, "units": 256, "unit_forget_bias": true, "use_bias": true, "recurrent_activation": "sigmoid", "activation": "tanh", "dtype": "float32", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "lstm_cell", "dropout": 0.05, "bias_initializer": {"class_name": "Zeros", "config": {}}, "recurrent_dropout": 0.05, "kernel_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "implementation": 1, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_constraint": null, "bias_regularizer": null, "recurrent_regularizer": null}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
)layer_metrics
trainable_variables
	variables
*non_trainable_variables
+metrics
,layer_regularization_losses
regularization_losses

-states

.layers
*c&call_and_return_all_conditional_losses
d__call__
&c"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

kernel
bias
/trainable_variables
0	keras_api
1	variables
2regularization_losses
*k&call_and_return_all_conditional_losses
l__call__"?
_tf_keras_layer?{"trainable": true, "dtype": "float32", "batch_input_shape": null, "expects_training_arg": false, "class_name": "Dense", "stateful": false, "build_input_shape": {"items": [null, 256], "class_name": "TensorShape"}, "input_spec": {"class_name": "InputSpec", "config": {"max_ndim": null, "axes": {"-1": 256}, "min_ndim": 2, "dtype": null, "shape": null, "ndim": null}}, "name": "mu_logstd_logmix_net", "config": {"trainable": true, "units": 482, "kernel_constraint": null, "activity_regularizer": null, "activation": "linear", "dtype": "float32", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "use_bias": true, "name": "mu_logstd_logmix_net", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_constraint": null, "bias_regularizer": null}}
.
0
1"
trackable_list_wrapper
?
3layer_metrics
trainable_variables
	variables
4non_trainable_variables
5metrics
6layer_regularization_losses
regularization_losses

7layers
*f&call_and_return_all_conditional_losses
g__call__
&f"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0:.	#?2dropout_lstm/lstm_cell/kernel
;:9
??2'dropout_lstm/lstm_cell/recurrent_kernel
*:(?2dropout_lstm/lstm_cell/bias
/:-
??2mu_logstd_logmix_net/kernel
(:&?2mu_logstd_logmix_net/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
<layer_metrics
%trainable_variables
'	variables
=non_trainable_variables
>metrics
?layer_regularization_losses
(regularization_losses

@layers
*i&call_and_return_all_conditional_losses
j__call__
&i"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Alayer_metrics
/trainable_variables
1	variables
Bnon_trainable_variables
Cmetrics
Dlayer_regularization_losses
2regularization_losses

Elayers
*k&call_and_return_all_conditional_losses
l__call__
&k"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
	Ftotal
	Gcount
H	keras_api
I	variables"?
_tf_keras_metricj{"dtype": "float32", "class_name": "Mean", "name": "loss", "config": {"dtype": "float32", "name": "loss"}}
?
	Jtotal
	Kcount
L	keras_api
M	variables"?
_tf_keras_metricr{"dtype": "float32", "class_name": "Mean", "name": "MDN_loss", "config": {"dtype": "float32", "name": "MDN_loss"}}
?
	Ntotal
	Ocount
P	keras_api
Q	variables"?
_tf_keras_metricn{"dtype": "float32", "class_name": "Mean", "name": "d_loss", "config": {"dtype": "float32", "name": "d_loss"}}
?
	Rtotal
	Scount
T	keras_api
U	variables"?
_tf_keras_metricn{"dtype": "float32", "class_name": "Mean", "name": "r_loss", "config": {"dtype": "float32", "name": "r_loss"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
-
I	variables"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
:  (2total
:  (2count
-
M	variables"
_generic_user_object
.
J0
K1"
trackable_list_wrapper
:  (2total
:  (2count
-
Q	variables"
_generic_user_object
.
N0
O1"
trackable_list_wrapper
:  (2total
:  (2count
-
U	variables"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
5:3	#?2$Adam/dropout_lstm/lstm_cell/kernel/m
@:>
??2.Adam/dropout_lstm/lstm_cell/recurrent_kernel/m
/:-?2"Adam/dropout_lstm/lstm_cell/bias/m
4:2
??2"Adam/mu_logstd_logmix_net/kernel/m
-:+?2 Adam/mu_logstd_logmix_net/bias/m
5:3	#?2$Adam/dropout_lstm/lstm_cell/kernel/v
@:>
??2.Adam/dropout_lstm/lstm_cell/recurrent_kernel/v
/:-?2"Adam/dropout_lstm/lstm_cell/bias/v
4:2
??2"Adam/mu_logstd_logmix_net/kernel/v
-:+?2 Adam/mu_logstd_logmix_net/bias/v
?2?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6426563
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6426843
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6427225
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6427505?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_mdnrnn_layer_call_fn_6427524
(__inference_mdnrnn_layer_call_fn_6426862
(__inference_mdnrnn_layer_call_fn_6427543
(__inference_mdnrnn_layer_call_fn_6426881?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_6424541?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_1??????????#
?2?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428125
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6427866
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428478
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428737?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_dropout_lstm_layer_call_fn_6428155
.__inference_dropout_lstm_layer_call_fn_6428140
.__inference_dropout_lstm_layer_call_fn_6428752
.__inference_dropout_lstm_layer_call_fn_6428767?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference__create_dropout_mask_134875
'__inference__create_dropout_mask_134925
'__inference__create_dropout_mask_135007?
???
FullArgSpec2
args*?'
jself
jinputs

jtraining
jcount
varargs
 
varkw
 
defaults?
`

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_6428787
G__inference_sequential_layer_call_and_return_conditional_losses_6425343
G__inference_sequential_layer_call_and_return_conditional_losses_6428777
G__inference_sequential_layer_call_and_return_conditional_losses_6425334?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_layer_call_fn_6428805
,__inference_sequential_layer_call_fn_6425380
,__inference_sequential_layer_call_fn_6428796
,__inference_sequential_layer_call_fn_6425362?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
4B2
%__inference_signature_wrapper_6426219input_1
?2?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_6429041
F__inference_lstm_cell_layer_call_and_return_conditional_losses_6428956?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lstm_cell_layer_call_fn_6428822
+__inference_lstm_cell_layer_call_fn_6428839?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_6429051?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_mu_logstd_logmix_net_layer_call_fn_6429060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
'__inference__create_dropout_mask_134875?7?4
-?*
 ?
inputs?????????#
p
`
? "w?t
?
0?????????#
?
1?????????#
?
2?????????#
?
3?????????#?
'__inference__create_dropout_mask_134925?7?4
-?*
 ?
inputs?????????#
p 
`
? "w?t
?
0?????????#
?
1?????????#
?
2?????????#
?
3?????????#?
'__inference__create_dropout_mask_135007?.?+
$?!
?
inputsd#
p
`
? "S?P
?
0d#
?
1d#
?
2d#
?
3d#?
"__inference__wrapped_model_6424541?5?2
+?(
&?#
input_1??????????#
? "n?k
%
MDN?
MDN??????????
 
d?
d?????????
 
r?
r??????????
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6427866?@?=
6?3
%?"
inputs??????????#

 
p

 
? "r?o
h?e
#? 
0/0???????????
?
0/1??????????
?
0/2??????????
? ?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428125?@?=
6?3
%?"
inputs??????????#

 
p 

 
? "r?o
h?e
#? 
0/0???????????
?
0/1??????????
?
0/2??????????
? ?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428478?O?L
E?B
4?1
/?,
inputs/0??????????????????#

 
p

 
? "z?w
p?m
+?(
0/0???????????????????
?
0/1??????????
?
0/2??????????
? ?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_6428737?O?L
E?B
4?1
/?,
inputs/0??????????????????#

 
p 

 
? "z?w
p?m
+?(
0/0???????????????????
?
0/1??????????
?
0/2??????????
? ?
.__inference_dropout_lstm_layer_call_fn_6428140?@?=
6?3
%?"
inputs??????????#

 
p

 
? "b?_
!?
0???????????
?
1??????????
?
2???????????
.__inference_dropout_lstm_layer_call_fn_6428155?@?=
6?3
%?"
inputs??????????#

 
p 

 
? "b?_
!?
0???????????
?
1??????????
?
2???????????
.__inference_dropout_lstm_layer_call_fn_6428752?O?L
E?B
4?1
/?,
inputs/0??????????????????#

 
p

 
? "j?g
)?&
0???????????????????
?
1??????????
?
2???????????
.__inference_dropout_lstm_layer_call_fn_6428767?O?L
E?B
4?1
/?,
inputs/0??????????????????#

 
p 

 
? "j?g
)?&
0???????????????????
?
1??????????
?
2???????????
F__inference_lstm_cell_layer_call_and_return_conditional_losses_6428956???
x?u
 ?
inputs?????????#
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_6429041???
x?u
 ?
inputs?????????#
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
+__inference_lstm_cell_layer_call_fn_6428822???
x?u
 ?
inputs?????????#
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
+__inference_lstm_cell_layer_call_fn_6428839???
x?u
 ?
inputs?????????#
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6426563?8?5
.?+
%?"
inputs??????????#
p
? "~?{
t?q
'
MDN ?
0/MDN??????????
"
d?
0/d?????????
"
r?
0/r?????????
? ?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6426843?8?5
.?+
%?"
inputs??????????#
p 
? "~?{
t?q
'
MDN ?
0/MDN??????????
"
d?
0/d?????????
"
r?
0/r?????????
? ?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6427225?9?6
/?,
&?#
input_1??????????#
p
? "~?{
t?q
'
MDN ?
0/MDN??????????
"
d?
0/d?????????
"
r?
0/r?????????
? ?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_6427505?9?6
/?,
&?#
input_1??????????#
p 
? "~?{
t?q
'
MDN ?
0/MDN??????????
"
d?
0/d?????????
"
r?
0/r?????????
? ?
(__inference_mdnrnn_layer_call_fn_6426862?8?5
.?+
%?"
inputs??????????#
p
? "n?k
%
MDN?
MDN??????????
 
d?
d?????????
 
r?
r??????????
(__inference_mdnrnn_layer_call_fn_6426881?8?5
.?+
%?"
inputs??????????#
p 
? "n?k
%
MDN?
MDN??????????
 
d?
d?????????
 
r?
r??????????
(__inference_mdnrnn_layer_call_fn_6427524?9?6
/?,
&?#
input_1??????????#
p
? "n?k
%
MDN?
MDN??????????
 
d?
d?????????
 
r?
r??????????
(__inference_mdnrnn_layer_call_fn_6427543?9?6
/?,
&?#
input_1??????????#
p 
? "n?k
%
MDN?
MDN??????????
 
d?
d?????????
 
r?
r??????????
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_6429051^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
6__inference_mu_logstd_logmix_net_layer_call_fn_6429060Q0?-
&?#
!?
inputs??????????
? "????????????
G__inference_sequential_layer_call_and_return_conditional_losses_6425334g9?6
/?,
"?
input_1??????????
p

 
? "&?#
?
0??????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_6425343g9?6
/?,
"?
input_1??????????
p 

 
? "&?#
?
0??????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_6428777f8?5
.?+
!?
inputs??????????
p

 
? "&?#
?
0??????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_6428787f8?5
.?+
!?
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
,__inference_sequential_layer_call_fn_6425362Z9?6
/?,
"?
input_1??????????
p

 
? "????????????
,__inference_sequential_layer_call_fn_6425380Z9?6
/?,
"?
input_1??????????
p 

 
? "????????????
,__inference_sequential_layer_call_fn_6428796Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
,__inference_sequential_layer_call_fn_6428805Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
%__inference_signature_wrapper_6426219?@?=
? 
6?3
1
input_1&?#
input_1??????????#"n?k
%
MDN?
MDN??????????
 
d?
d?????????
 
r?
r?????????