??7
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
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??6
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
shape:	A?*.
shared_namedropout_lstm/lstm_cell/kernel
?
1dropout_lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpdropout_lstm/lstm_cell/kernel*
_output_shapes
:	A?*
dtype0
?
'dropout_lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*8
shared_name)'dropout_lstm/lstm_cell/recurrent_kernel
?
;dropout_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp'dropout_lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
dropout_lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namedropout_lstm/lstm_cell/bias
?
/dropout_lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpdropout_lstm/lstm_cell/bias*
_output_shapes	
:?*
dtype0
?
mu_logstd_logmix_net/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namemu_logstd_logmix_net/kernel
?
/mu_logstd_logmix_net/kernel/Read/ReadVariableOpReadVariableOpmu_logstd_logmix_net/kernel* 
_output_shapes
:
??*
dtype0
?
mu_logstd_logmix_net/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namemu_logstd_logmix_net/bias
?
-mu_logstd_logmix_net/bias/Read/ReadVariableOpReadVariableOpmu_logstd_logmix_net/bias*
_output_shapes	
:?*
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
?
$Adam/dropout_lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	A?*5
shared_name&$Adam/dropout_lstm/lstm_cell/kernel/m
?
8Adam/dropout_lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/dropout_lstm/lstm_cell/kernel/m*
_output_shapes
:	A?*
dtype0
?
.Adam/dropout_lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*?
shared_name0.Adam/dropout_lstm/lstm_cell/recurrent_kernel/m
?
BAdam/dropout_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp.Adam/dropout_lstm/lstm_cell/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
"Adam/dropout_lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/dropout_lstm/lstm_cell/bias/m
?
6Adam/dropout_lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOp"Adam/dropout_lstm/lstm_cell/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/mu_logstd_logmix_net/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/mu_logstd_logmix_net/kernel/m
?
6Adam/mu_logstd_logmix_net/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/mu_logstd_logmix_net/kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/mu_logstd_logmix_net/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/mu_logstd_logmix_net/bias/m
?
4Adam/mu_logstd_logmix_net/bias/m/Read/ReadVariableOpReadVariableOp Adam/mu_logstd_logmix_net/bias/m*
_output_shapes	
:?*
dtype0
?
$Adam/dropout_lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	A?*5
shared_name&$Adam/dropout_lstm/lstm_cell/kernel/v
?
8Adam/dropout_lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/dropout_lstm/lstm_cell/kernel/v*
_output_shapes
:	A?*
dtype0
?
.Adam/dropout_lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*?
shared_name0.Adam/dropout_lstm/lstm_cell/recurrent_kernel/v
?
BAdam/dropout_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp.Adam/dropout_lstm/lstm_cell/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
"Adam/dropout_lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/dropout_lstm/lstm_cell/bias/v
?
6Adam/dropout_lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOp"Adam/dropout_lstm/lstm_cell/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/mu_logstd_logmix_net/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/mu_logstd_logmix_net/kernel/v
?
6Adam/mu_logstd_logmix_net/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/mu_logstd_logmix_net/kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/mu_logstd_logmix_net/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/mu_logstd_logmix_net/bias/v
?
4Adam/mu_logstd_logmix_net/bias/v/Read/ReadVariableOpReadVariableOp Adam/mu_logstd_logmix_net/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?'
value?'B?' B?'
?
	optimizer
loss_fn
inference_base
out_net
loss

signatures
regularization_losses
trainable_variables
		keras_api

	variables
?
iter

beta_1

beta_2
	decay
learning_ratemQmRmSmTmUvVvWvXvYvZ
 
l
cell

state_spec
regularization_losses
trainable_variables
	keras_api
	variables
y
layer_with_weights-0
layer-0
regularization_losses
trainable_variables
	keras_api
	variables
 
 
 
#
0
1
2
3
4
?
 non_trainable_variables
!layer_regularization_losses
regularization_losses

"layers
#layer_metrics
trainable_variables

	variables
$metrics
#
0
1
2
3
4
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
%regularization_losses
&trainable_variables
'	keras_api
(	variables
 
 

0
1
2
?
)non_trainable_variables
*layer_regularization_losses

+states
regularization_losses

,layers
-layer_metrics
trainable_variables
	variables
.metrics

0
1
2
h

kernel
bias
/regularization_losses
0trainable_variables
1	keras_api
2	variables
 

0
1
?
3non_trainable_variables
4layer_regularization_losses
regularization_losses

5layers
6layer_metrics
trainable_variables
	variables
7metrics

0
1
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

0
1
 

80
91
:2
 

0
1
2
?
;non_trainable_variables
<layer_regularization_losses
%regularization_losses

=layers
>layer_metrics
&trainable_variables
(	variables
?metrics

0
1
2
 
 
 

0
 
 
 

0
1
?
@non_trainable_variables
Alayer_regularization_losses
/regularization_losses

Blayers
Clayer_metrics
0trainable_variables
2	variables
Dmetrics

0
1
 
 

0
 
 
4
	Etotal
	Fcount
G	variables
H	keras_api
4
	Itotal
	Jcount
K	variables
L	keras_api
4
	Mtotal
	Ncount
O	variables
P	keras_api
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

E0
F1

G	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

K	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

O	variables
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
:??????????A*
dtype0*!
shape:??????????A
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dropout_lstm/lstm_cell/kerneldropout_lstm/lstm_cell/bias'dropout_lstm/lstm_cell/recurrent_kernelmu_logstd_logmix_net/kernelmu_logstd_logmix_net/bias*
Tin

2*
Tout
2*;
_output_shapes)
':??????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*.
f)R'
%__inference_signature_wrapper_4346067
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp1dropout_lstm/lstm_cell/kernel/Read/ReadVariableOp;dropout_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp/dropout_lstm/lstm_cell/bias/Read/ReadVariableOp/mu_logstd_logmix_net/kernel/Read/ReadVariableOp-mu_logstd_logmix_net/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp8Adam/dropout_lstm/lstm_cell/kernel/m/Read/ReadVariableOpBAdam/dropout_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp6Adam/dropout_lstm/lstm_cell/bias/m/Read/ReadVariableOp6Adam/mu_logstd_logmix_net/kernel/m/Read/ReadVariableOp4Adam/mu_logstd_logmix_net/bias/m/Read/ReadVariableOp8Adam/dropout_lstm/lstm_cell/kernel/v/Read/ReadVariableOpBAdam/dropout_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp6Adam/dropout_lstm/lstm_cell/bias/v/Read/ReadVariableOp6Adam/mu_logstd_logmix_net/kernel/v/Read/ReadVariableOp4Adam/mu_logstd_logmix_net/bias/v/Read/ReadVariableOpConst*'
Tin 
2	*
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
 __inference__traced_save_4349976
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedropout_lstm/lstm_cell/kernel'dropout_lstm/lstm_cell/recurrent_kerneldropout_lstm/lstm_cell/biasmu_logstd_logmix_net/kernelmu_logstd_logmix_net/biastotalcounttotal_1count_1total_2count_2$Adam/dropout_lstm/lstm_cell/kernel/m.Adam/dropout_lstm/lstm_cell/recurrent_kernel/m"Adam/dropout_lstm/lstm_cell/bias/m"Adam/mu_logstd_logmix_net/kernel/m Adam/mu_logstd_logmix_net/bias/m$Adam/dropout_lstm/lstm_cell/kernel/v.Adam/dropout_lstm/lstm_cell/recurrent_kernel/v"Adam/dropout_lstm/lstm_cell/bias/v"Adam/mu_logstd_logmix_net/kernel/v Adam/mu_logstd_logmix_net/bias/v*&
Tin
2*
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
#__inference__traced_restore_4350066ñ5
?
?
while_cond_4348857
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_4348857___redundant_placeholder0/
+while_cond_4348857___redundant_placeholder1/
+while_cond_4348857___redundant_placeholder2/
+while_cond_4348857___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
¦
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4349100
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
B :?2
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
B :?2
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2	
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
 :??????????????????A2
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
valueB"????A   27
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
:?????????A*
shrink_axis_mask2
strided_slice_2?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlicestrided_slice_2:output:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2??<22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/dropout/Mul_1:z:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/dropout_1/Mul_1:z:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/dropout_2/Mul_1:z:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/dropout_3/Mul_1:z:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3x
lstm_cell/ones_like_5/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??#22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4348858*
condR
while_cond_4348857*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????A:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????A
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
+__inference_lstm_cell_layer_call_fn_4349522

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
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_43443182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????A:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
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
??
?
dropout_lstm_while_body_4343892#
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
ـ
?
dropout_lstm_while_body_4347159#
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2ؗ?20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/dropout/Mul_1:z:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/dropout_1/Mul_1:z:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/dropout_2/Mul_1:z:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/dropout_3/Mul_1:z:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2랧22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??j22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??v22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?
while_body_4348427
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
G__inference_sequential_layer_call_and_return_conditional_losses_4345027

inputs 
mu_logstd_logmix_net_4345021 
mu_logstd_logmix_net_4345023
identity??,mu_logstd_logmix_net/StatefulPartitionedCall?
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinputsmu_logstd_logmix_net_4345021mu_logstd_logmix_net_4345023*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_43449712.
,mu_logstd_logmix_net/StatefulPartitionedCall?
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0-^mu_logstd_logmix_net/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?E
?
 __inference__traced_save_4349976
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
"savev2_count_2_read_readvariableopC
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
value3B1 B+_temp_d37b43e9d25d432fa64acc1d8f6a1db4/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop8savev2_dropout_lstm_lstm_cell_kernel_read_readvariableopBsavev2_dropout_lstm_lstm_cell_recurrent_kernel_read_readvariableop6savev2_dropout_lstm_lstm_cell_bias_read_readvariableop6savev2_mu_logstd_logmix_net_kernel_read_readvariableop4savev2_mu_logstd_logmix_net_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop?savev2_adam_dropout_lstm_lstm_cell_kernel_m_read_readvariableopIsavev2_adam_dropout_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop=savev2_adam_dropout_lstm_lstm_cell_bias_m_read_readvariableop=savev2_adam_mu_logstd_logmix_net_kernel_m_read_readvariableop;savev2_adam_mu_logstd_logmix_net_bias_m_read_readvariableop?savev2_adam_dropout_lstm_lstm_cell_kernel_v_read_readvariableopIsavev2_adam_dropout_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop=savev2_adam_dropout_lstm_lstm_cell_bias_v_read_readvariableop=savev2_adam_mu_logstd_logmix_net_kernel_v_read_readvariableop;savev2_adam_mu_logstd_logmix_net_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
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
?: : : : : : :	A?:
??:?:
??:?: : : : : : :	A?:
??:?:
??:?:	A?:
??:?:
??:?: 2(
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
:	A?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:
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
: :%!

_output_shapes
:	A?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	A?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?
?
.__inference_dropout_lstm_layer_call_fn_4348635

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
A:???????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_43458402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????A:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????A
 
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
??
?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_4349727

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2?{
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
:?????????@*

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
:?????????@2
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
:?????????@2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???2&
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
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
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
:?????????@2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???2(
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
:?????????@2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
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
:?????????@2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
:?????????@2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
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
:?????????@2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2ϣ)2(
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
:?????????@2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????A2
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
:?????????A2

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
:?????????A2

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
:?????????A2

concat_3^
ones_like_5/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_5/Shapek
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_5/Const?
ones_like_5Fillones_like_5/Shape:output:0ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_5g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_4/Const?
dropout_4/MulMulones_like_5:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_5:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?է2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_5/Const?
dropout_5/MulMulones_like_5:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_5:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ȫS2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_6/Const?
dropout_6/MulMulones_like_5:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_5:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_7/Const?
dropout_7/MulMulones_like_5:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_5:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mul_1\
mulMulinputsconcat:output:0*
T0*'
_output_shapes
:?????????A2
mulb
mul_1Mulinputsconcat_1:output:0*
T0*'
_output_shapes
:?????????A2
mul_1b
mul_2Mulinputsconcat_2:output:0*
T0*'
_output_shapes
:?????????A2
mul_2b
mul_3Mulinputsconcat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

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
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3g
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4g
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5g
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6g
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_5v
MatMul_4MatMul	mul_4:z:0strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_1:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_6v
MatMul_5MatMul	mul_5:z:0strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_2:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_7v
MatMul_6MatMul	mul_6:z:0strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_3:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_8v
MatMul_7MatMul	mul_7:z:0strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????A:??????????:??????????::::O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
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
?u
?
#__inference__traced_restore_4350066
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
assignvariableop_15_count_2<
8assignvariableop_16_adam_dropout_lstm_lstm_cell_kernel_mF
Bassignvariableop_17_adam_dropout_lstm_lstm_cell_recurrent_kernel_m:
6assignvariableop_18_adam_dropout_lstm_lstm_cell_bias_m:
6assignvariableop_19_adam_mu_logstd_logmix_net_kernel_m8
4assignvariableop_20_adam_mu_logstd_logmix_net_bias_m<
8assignvariableop_21_adam_dropout_lstm_lstm_cell_kernel_vF
Bassignvariableop_22_adam_dropout_lstm_lstm_cell_recurrent_kernel_v:
6assignvariableop_23_adam_dropout_lstm_lstm_cell_bias_v:
6assignvariableop_24_adam_mu_logstd_logmix_net_kernel_v8
4assignvariableop_25_adam_mu_logstd_logmix_net_bias_v
identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
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
AssignVariableOp_16AssignVariableOp8assignvariableop_16_adam_dropout_lstm_lstm_cell_kernel_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpBassignvariableop_17_adam_dropout_lstm_lstm_cell_recurrent_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_dropout_lstm_lstm_cell_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_mu_logstd_logmix_net_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_mu_logstd_logmix_net_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp8assignvariableop_21_adam_dropout_lstm_lstm_cell_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpBassignvariableop_22_adam_dropout_lstm_lstm_cell_recurrent_kernel_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_dropout_lstm_lstm_cell_bias_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_mu_logstd_logmix_net_kernel_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_mu_logstd_logmix_net_bias_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25?
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
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26?
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_27"#
identity_27Identity_27:output:0*}
_input_shapesl
j: ::::::::::::::::::::::::::2$
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
AssignVariableOp_25AssignVariableOp_252(
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
: 
?

?
G__inference_sequential_layer_call_and_return_conditional_losses_4349505

inputs7
3mu_logstd_logmix_net_matmul_readvariableop_resource8
4mu_logstd_logmix_net_biasadd_readvariableop_resource
identity??
*mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp3mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*mu_logstd_logmix_net/MatMul/ReadVariableOp?
mu_logstd_logmix_net/MatMulMatMulinputs2mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mu_logstd_logmix_net/MatMul?
+mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp4mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
mu_logstd_logmix_net/BiasAddBiasAdd%mu_logstd_logmix_net/MatMul:product:03mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mu_logstd_logmix_net/BiasAddz
IdentityIdentity%mu_logstd_logmix_net/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_4348025
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_4348025___redundant_placeholder0/
+while_cond_4348025___redundant_placeholder1/
+while_cond_4348025___redundant_placeholder2/
+while_cond_4348025___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4348268

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
B :?2
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
B :?2
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2	
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
:??????????A2
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
valueB"????A   27
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
:?????????A*
shrink_axis_mask2
strided_slice_2?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlicestrided_slice_2:output:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2أ20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2??u22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2ޟ?22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2??22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/dropout/Mul_1:z:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/dropout_1/Mul_1:z:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/dropout_2/Mul_1:z:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/dropout_3/Mul_1:z:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3x
lstm_cell/ones_like_5/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??022
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??R22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4348026*
condR
while_cond_4348025*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
:???????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????A:::2
whilewhile:T P
,
_output_shapes
:??????????A
 
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
??
?
while_body_4348858
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2?׼20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2?٨22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2хN22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/dropout/Mul_1:z:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/dropout_1/Mul_1:z:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/dropout_2/Mul_1:z:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/dropout_3/Mul_1:z:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??W22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2̗?22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4345503

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
B :?2
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
B :?2
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2	
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
:??????????A2
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
valueB"????A   27
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
:?????????A*
shrink_axis_mask2
strided_slice_2?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlicestrided_slice_2:output:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2≯20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2?22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2ʙ?22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/dropout/Mul_1:z:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/dropout_1/Mul_1:z:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/dropout_2/Mul_1:z:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/dropout_3/Mul_1:z:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3x
lstm_cell/ones_like_5/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??\22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ח22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2͢?22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??!22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4345261*
condR
while_cond_4345260*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
:???????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????A:::2
whilewhile:T P
,
_output_shapes
:??????????A
 
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
??
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4347416

inputs8
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1??dropout_lstm/while|
dropout_lstm/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:??????????A2
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
B :?2
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
B :?2
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
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2
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
:??????????A2
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
valueB"????A   2D
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
:?????????A*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
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
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice%dropout_lstm/strided_slice_2:output:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2&
$dropout_lstm/lstm_cell/strided_slice?
&dropout_lstm/lstm_cell/ones_like/ShapeShape-dropout_lstm/lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2"
 dropout_lstm/lstm_cell/ones_like?
$dropout_lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2&
$dropout_lstm/lstm_cell/dropout/Const?
"dropout_lstm/lstm_cell/dropout/MulMul)dropout_lstm/lstm_cell/ones_like:output:0-dropout_lstm/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2$
"dropout_lstm/lstm_cell/dropout/Mul?
$dropout_lstm/lstm_cell/dropout/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$dropout_lstm/lstm_cell/dropout/Shape?
;dropout_lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform-dropout_lstm/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???2=
;dropout_lstm/lstm_cell/dropout/random_uniform/RandomUniform?
-dropout_lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2/
-dropout_lstm/lstm_cell/dropout/GreaterEqual/y?
+dropout_lstm/lstm_cell/dropout/GreaterEqualGreaterEqualDdropout_lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:06dropout_lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2-
+dropout_lstm/lstm_cell/dropout/GreaterEqual?
#dropout_lstm/lstm_cell/dropout/CastCast/dropout_lstm/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2%
#dropout_lstm/lstm_cell/dropout/Cast?
$dropout_lstm/lstm_cell/dropout/Mul_1Mul&dropout_lstm/lstm_cell/dropout/Mul:z:0'dropout_lstm/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2&
$dropout_lstm/lstm_cell/dropout/Mul_1?
&dropout_lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_1/Const?
$dropout_lstm/lstm_cell/dropout_1/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2&
$dropout_lstm/lstm_cell/dropout_1/Mul?
&dropout_lstm/lstm_cell/dropout_1/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_1/Shape?
=dropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_1/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2/
-dropout_lstm/lstm_cell/dropout_1/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_1/CastCast1dropout_lstm/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2'
%dropout_lstm/lstm_cell/dropout_1/Cast?
&dropout_lstm/lstm_cell/dropout_1/Mul_1Mul(dropout_lstm/lstm_cell/dropout_1/Mul:z:0)dropout_lstm/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2(
&dropout_lstm/lstm_cell/dropout_1/Mul_1?
&dropout_lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_2/Const?
$dropout_lstm/lstm_cell/dropout_2/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2&
$dropout_lstm/lstm_cell/dropout_2/Mul?
&dropout_lstm/lstm_cell/dropout_2/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_2/Shape?
=dropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2/
-dropout_lstm/lstm_cell/dropout_2/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_2/CastCast1dropout_lstm/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2'
%dropout_lstm/lstm_cell/dropout_2/Cast?
&dropout_lstm/lstm_cell/dropout_2/Mul_1Mul(dropout_lstm/lstm_cell/dropout_2/Mul:z:0)dropout_lstm/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2(
&dropout_lstm/lstm_cell/dropout_2/Mul_1?
&dropout_lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_3/Const?
$dropout_lstm/lstm_cell/dropout_3/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2&
$dropout_lstm/lstm_cell/dropout_3/Mul?
&dropout_lstm/lstm_cell/dropout_3/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_3/Shape?
=dropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_3/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2/
-dropout_lstm/lstm_cell/dropout_3/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_3/CastCast1dropout_lstm/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2'
%dropout_lstm/lstm_cell/dropout_3/Cast?
&dropout_lstm/lstm_cell/dropout_3/Mul_1Mul(dropout_lstm/lstm_cell/dropout_3/Mul:z:0)dropout_lstm/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2(
&dropout_lstm/lstm_cell/dropout_3/Mul_1?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
(dropout_lstm/lstm_cell/ones_like_1/ShapeShape/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_1/Shape?
(dropout_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_1/Const?
"dropout_lstm/lstm_cell/ones_like_1Fill1dropout_lstm/lstm_cell/ones_like_1/Shape:output:01dropout_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_1?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
(dropout_lstm/lstm_cell/ones_like_2/ShapeShape/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_2/Shape?
(dropout_lstm/lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_2/Const?
"dropout_lstm/lstm_cell/ones_like_2Fill1dropout_lstm/lstm_cell/ones_like_2/Shape:output:01dropout_lstm/lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_2?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
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
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
(dropout_lstm/lstm_cell/ones_like_3/ShapeShape/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_3/Shape?
(dropout_lstm/lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_3/Const?
"dropout_lstm/lstm_cell/ones_like_3Fill1dropout_lstm/lstm_cell/ones_like_3/Shape:output:01dropout_lstm/lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_3?
,dropout_lstm/lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_4/stack?
.dropout_lstm/lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_4/stack_1?
.dropout_lstm/lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_4/stack_2?
&dropout_lstm/lstm_cell/strided_slice_4StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_4/stack:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_4?
(dropout_lstm/lstm_cell/ones_like_4/ShapeShape/dropout_lstm/lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_4/Shape?
(dropout_lstm/lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_4/Const?
"dropout_lstm/lstm_cell/ones_like_4Fill1dropout_lstm/lstm_cell/ones_like_4/Shape:output:01dropout_lstm/lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_4?
"dropout_lstm/lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"dropout_lstm/lstm_cell/concat/axis?
dropout_lstm/lstm_cell/concatConcatV2(dropout_lstm/lstm_cell/dropout/Mul_1:z:0+dropout_lstm/lstm_cell/ones_like_1:output:0+dropout_lstm/lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/concat?
$dropout_lstm/lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_1/axis?
dropout_lstm/lstm_cell/concat_1ConcatV2*dropout_lstm/lstm_cell/dropout_1/Mul_1:z:0+dropout_lstm/lstm_cell/ones_like_2:output:0-dropout_lstm/lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_1?
$dropout_lstm/lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_2/axis?
dropout_lstm/lstm_cell/concat_2ConcatV2*dropout_lstm/lstm_cell/dropout_2/Mul_1:z:0+dropout_lstm/lstm_cell/ones_like_3:output:0-dropout_lstm/lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_2?
$dropout_lstm/lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_3/axis?
dropout_lstm/lstm_cell/concat_3ConcatV2*dropout_lstm/lstm_cell/dropout_3/Mul_1:z:0+dropout_lstm/lstm_cell/ones_like_4:output:0-dropout_lstm/lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_3?
(dropout_lstm/lstm_cell/ones_like_5/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_5/Shape?
(dropout_lstm/lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_5/Const?
"dropout_lstm/lstm_cell/ones_like_5Fill1dropout_lstm/lstm_cell/ones_like_5/Shape:output:01dropout_lstm/lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2$
"dropout_lstm/lstm_cell/ones_like_5?
&dropout_lstm/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_4/Const?
$dropout_lstm/lstm_cell/dropout_4/MulMul+dropout_lstm/lstm_cell/ones_like_5:output:0/dropout_lstm/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_4/Mul?
&dropout_lstm/lstm_cell/dropout_4/ShapeShape+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_4/Shape?
=dropout_lstm/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ɣ2?
=dropout_lstm/lstm_cell/dropout_4/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_4/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_4/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_4/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_4/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_4/CastCast1dropout_lstm/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_4/Cast?
&dropout_lstm/lstm_cell/dropout_4/Mul_1Mul(dropout_lstm/lstm_cell/dropout_4/Mul:z:0)dropout_lstm/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_4/Mul_1?
&dropout_lstm/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_5/Const?
$dropout_lstm/lstm_cell/dropout_5/MulMul+dropout_lstm/lstm_cell/ones_like_5:output:0/dropout_lstm/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_5/Mul?
&dropout_lstm/lstm_cell/dropout_5/ShapeShape+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_5/Shape?
=dropout_lstm/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_5/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_5/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_5/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_5/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_5/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_5/CastCast1dropout_lstm/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_5/Cast?
&dropout_lstm/lstm_cell/dropout_5/Mul_1Mul(dropout_lstm/lstm_cell/dropout_5/Mul:z:0)dropout_lstm/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_5/Mul_1?
&dropout_lstm/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_6/Const?
$dropout_lstm/lstm_cell/dropout_6/MulMul+dropout_lstm/lstm_cell/ones_like_5:output:0/dropout_lstm/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_6/Mul?
&dropout_lstm/lstm_cell/dropout_6/ShapeShape+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_6/Shape?
=dropout_lstm/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??h2?
=dropout_lstm/lstm_cell/dropout_6/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_6/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_6/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_6/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_6/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_6/CastCast1dropout_lstm/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_6/Cast?
&dropout_lstm/lstm_cell/dropout_6/Mul_1Mul(dropout_lstm/lstm_cell/dropout_6/Mul:z:0)dropout_lstm/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_6/Mul_1?
&dropout_lstm/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_7/Const?
$dropout_lstm/lstm_cell/dropout_7/MulMul+dropout_lstm/lstm_cell/ones_like_5:output:0/dropout_lstm/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_7/Mul?
&dropout_lstm/lstm_cell/dropout_7/ShapeShape+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_7/Shape?
=dropout_lstm/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ɫ2?
=dropout_lstm/lstm_cell/dropout_7/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_7/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_7/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_7/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_7/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_7/CastCast1dropout_lstm/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_7/Cast?
&dropout_lstm/lstm_cell/dropout_7/Mul_1Mul(dropout_lstm/lstm_cell/dropout_7/Mul:z:0)dropout_lstm/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_7/Mul_1?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:0&dropout_lstm/lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
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
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
,dropout_lstm/lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,dropout_lstm/lstm_cell/strided_slice_5/stack?
.dropout_lstm/lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_5/stack_1?
.dropout_lstm/lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_5/stack_2?
&dropout_lstm/lstm_cell/strided_slice_5StridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:05dropout_lstm/lstm_cell/strided_slice_5/stack:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_5?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0/dropout_lstm/lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_6/stack?
.dropout_lstm/lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_6/stack_1?
.dropout_lstm/lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_6/stack_2?
&dropout_lstm/lstm_cell/strided_slice_6StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_6/stack:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_6?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_7/stack?
.dropout_lstm/lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_7/stack_1?
.dropout_lstm/lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_7/stack_2?
&dropout_lstm/lstm_cell/strided_slice_7StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_7/stack:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_7?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_8/stack?
.dropout_lstm/lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_8/stack_1?
.dropout_lstm/lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_8/stack_2?
&dropout_lstm/lstm_cell/strided_slice_8StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_8/stack:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_8?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_4347159*+
cond#R!
dropout_lstm_while_cond_4347158*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
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
valueB"    ?  2
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
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
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
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
IdentityIdentitystrided_slice:output:0^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_1:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::2(
dropout_lstm/whiledropout_lstm/while:T P
,
_output_shapes
:??????????A
 
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
dropout_lstm_while_cond_4347575#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_4347575___redundant_placeholder0<
8dropout_lstm_while_cond_4347575___redundant_placeholder1<
8dropout_lstm_while_cond_4347575___redundant_placeholder2<
8dropout_lstm_while_cond_4347575___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?
dropout_lstm_while_cond_4343891#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_4343891___redundant_placeholder0<
8dropout_lstm_while_cond_4343891___redundant_placeholder1<
8dropout_lstm_while_cond_4343891___redundant_placeholder2<
8dropout_lstm_while_cond_4343891___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
(__inference_mdnrnn_layer_call_fn_4346918
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*;
_output_shapes)
':??????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_mdnrnn_layer_call_and_return_conditional_losses_43459782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????A
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
ܘ
?
"__inference__wrapped_model_4344085
input_18
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1??dropout_lstm/while}
dropout_lstm/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:??????????A2
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
B :?2
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
B :?2
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
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2
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
:??????????A2
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
valueB"????A   2D
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
:?????????A*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
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
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice%dropout_lstm/strided_slice_2:output:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2&
$dropout_lstm/lstm_cell/strided_slice?
&dropout_lstm/lstm_cell/ones_like/ShapeShape-dropout_lstm/lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2"
 dropout_lstm/lstm_cell/ones_like?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
(dropout_lstm/lstm_cell/ones_like_1/ShapeShape/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_1/Shape?
(dropout_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_1/Const?
"dropout_lstm/lstm_cell/ones_like_1Fill1dropout_lstm/lstm_cell/ones_like_1/Shape:output:01dropout_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_1?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
(dropout_lstm/lstm_cell/ones_like_2/ShapeShape/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_2/Shape?
(dropout_lstm/lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_2/Const?
"dropout_lstm/lstm_cell/ones_like_2Fill1dropout_lstm/lstm_cell/ones_like_2/Shape:output:01dropout_lstm/lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_2?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
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
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
(dropout_lstm/lstm_cell/ones_like_3/ShapeShape/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_3/Shape?
(dropout_lstm/lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_3/Const?
"dropout_lstm/lstm_cell/ones_like_3Fill1dropout_lstm/lstm_cell/ones_like_3/Shape:output:01dropout_lstm/lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_3?
,dropout_lstm/lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_4/stack?
.dropout_lstm/lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_4/stack_1?
.dropout_lstm/lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_4/stack_2?
&dropout_lstm/lstm_cell/strided_slice_4StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_4/stack:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_4?
(dropout_lstm/lstm_cell/ones_like_4/ShapeShape/dropout_lstm/lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_4/Shape?
(dropout_lstm/lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_4/Const?
"dropout_lstm/lstm_cell/ones_like_4Fill1dropout_lstm/lstm_cell/ones_like_4/Shape:output:01dropout_lstm/lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_4?
"dropout_lstm/lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"dropout_lstm/lstm_cell/concat/axis?
dropout_lstm/lstm_cell/concatConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_1:output:0+dropout_lstm/lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/concat?
$dropout_lstm/lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_1/axis?
dropout_lstm/lstm_cell/concat_1ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_2:output:0-dropout_lstm/lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_1?
$dropout_lstm/lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_2/axis?
dropout_lstm/lstm_cell/concat_2ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_3:output:0-dropout_lstm/lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_2?
$dropout_lstm/lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_3/axis?
dropout_lstm/lstm_cell/concat_3ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_4:output:0-dropout_lstm/lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_3?
(dropout_lstm/lstm_cell/ones_like_5/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_5/Shape?
(dropout_lstm/lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_5/Const?
"dropout_lstm/lstm_cell/ones_like_5Fill1dropout_lstm/lstm_cell/ones_like_5/Shape:output:01dropout_lstm/lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2$
"dropout_lstm/lstm_cell/ones_like_5?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:0&dropout_lstm/lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
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
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
,dropout_lstm/lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,dropout_lstm/lstm_cell/strided_slice_5/stack?
.dropout_lstm/lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_5/stack_1?
.dropout_lstm/lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_5/stack_2?
&dropout_lstm/lstm_cell/strided_slice_5StridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:05dropout_lstm/lstm_cell/strided_slice_5/stack:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_5?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0/dropout_lstm/lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_6/stack?
.dropout_lstm/lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_6/stack_1?
.dropout_lstm/lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_6/stack_2?
&dropout_lstm/lstm_cell/strided_slice_6StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_6/stack:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_6?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_7/stack?
.dropout_lstm/lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_7/stack_1?
.dropout_lstm/lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_7/stack_2?
&dropout_lstm/lstm_cell/strided_slice_7StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_7/stack:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_7?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_8/stack?
.dropout_lstm/lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_8/stack_1?
.dropout_lstm/lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_8/stack_2?
&dropout_lstm/lstm_cell/strided_slice_8StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_8/stack:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_8?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_4343892*+
cond#R!
dropout_lstm_while_cond_4343891*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
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
valueB"    ?  2
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
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
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
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
IdentityIdentitystrided_slice:output:0^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_1:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::2(
dropout_lstm/whiledropout_lstm/while:U Q
,
_output_shapes
:??????????A
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
G__inference_sequential_layer_call_and_return_conditional_losses_4345009

inputs 
mu_logstd_logmix_net_4345003 
mu_logstd_logmix_net_4345005
identity??,mu_logstd_logmix_net/StatefulPartitionedCall?
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinputsmu_logstd_logmix_net_4345003mu_logstd_logmix_net_4345005*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_43449712.
,mu_logstd_logmix_net/StatefulPartitionedCall?
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0-^mu_logstd_logmix_net/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_4349258
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_4349258___redundant_placeholder0/
+while_cond_4349258___redundant_placeholder1/
+while_cond_4349258___redundant_placeholder2/
+while_cond_4349258___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
.__inference_dropout_lstm_layer_call_fn_4349467
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
I:???????????????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_43449442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????A:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????A
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
?
while_body_4344873
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_4344897_0
lstm_cell_4344899_0
lstm_cell_4344901_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_4344897
lstm_cell_4344899
lstm_cell_4344901??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_4344897_0lstm_cell_4344899_0lstm_cell_4344901_0*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_43444592#
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
:??????????2

Identity_4?

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"(
lstm_cell_4344897lstm_cell_4344897_0"(
lstm_cell_4344899lstm_cell_4344899_0"(
lstm_cell_4344901lstm_cell_4344901_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
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
:??????????:.*
(
_output_shapes
:??????????:
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
+__inference_lstm_cell_layer_call_fn_4349539

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
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_43444592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????A:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
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
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4345978

inputs
dropout_lstm_4345952
dropout_lstm_4345954
dropout_lstm_4345956
sequential_4345963
sequential_4345965
identity

identity_1??$dropout_lstm/StatefulPartitionedCall?"sequential/StatefulPartitionedCall|
dropout_lstm/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:??????????A2
dropout_lstm/Cast?
$dropout_lstm/StatefulPartitionedCallStatefulPartitionedCalldropout_lstm/Cast:y:0dropout_lstm_4345952dropout_lstm_4345954dropout_lstm_4345956*
Tin
2*
Tout
2*U
_output_shapesC
A:???????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_43455032&
$dropout_lstm/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshape-dropout_lstm/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
"sequential/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0sequential_4345963sequential_4345965*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_43450092$
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
valueB"    ?  2
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
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
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
strided_slice_1StridedSlice+sequential/StatefulPartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
IdentityIdentitystrided_slice:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_1:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::2L
$dropout_lstm/StatefulPartitionedCall$dropout_lstm/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
,
_output_shapes
:??????????A
 
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
?w
?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_4344459

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2?{
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
:?????????@*

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
:?????????@2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????A2
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
:?????????A2

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
:?????????A2

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
:?????????A2

concat_3\
ones_like_5/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_5/Shapek
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_5/Const?
ones_like_5Fillones_like_5/Shape:output:0ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_5\
mulMulinputsconcat:output:0*
T0*'
_output_shapes
:?????????A2
mulb
mul_1Mulinputsconcat_1:output:0*
T0*'
_output_shapes
:?????????A2
mul_1b
mul_2Mulinputsconcat_2:output:0*
T0*'
_output_shapes
:?????????A2
mul_2b
mul_3Mulinputsconcat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

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
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3f
mul_4Mulstatesones_like_5:output:0*
T0*(
_output_shapes
:??????????2
mul_4f
mul_5Mulstatesones_like_5:output:0*
T0*(
_output_shapes
:??????????2
mul_5f
mul_6Mulstatesones_like_5:output:0*
T0*(
_output_shapes
:??????????2
mul_6f
mul_7Mulstatesones_like_5:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_5v
MatMul_4MatMul	mul_4:z:0strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_1:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_6v
MatMul_5MatMul	mul_5:z:0strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_2:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_7v
MatMul_6MatMul	mul_6:z:0strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_3:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_8v
MatMul_7MatMul	mul_7:z:0strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????A:??????????:??????????::::O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
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
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4345840

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
B :?2
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
B :?2
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2	
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
:??????????A2
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
valueB"????A   27
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
:?????????A*
shrink_axis_mask2
strided_slice_2?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlicestrided_slice_2:output:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3x
lstm_cell/ones_like_5/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4345662*
condR
while_cond_4345661*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
:???????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????A:::2
whilewhile:T P
,
_output_shapes
:??????????A
 
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
?
?
,__inference_sequential_layer_call_fn_4349485

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
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_43450272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4346548
input_18
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1??dropout_lstm/while}
dropout_lstm/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:??????????A2
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
B :?2
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
B :?2
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
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2
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
:??????????A2
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
valueB"????A   2D
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
:?????????A*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
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
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice%dropout_lstm/strided_slice_2:output:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2&
$dropout_lstm/lstm_cell/strided_slice?
&dropout_lstm/lstm_cell/ones_like/ShapeShape-dropout_lstm/lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2"
 dropout_lstm/lstm_cell/ones_like?
$dropout_lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2&
$dropout_lstm/lstm_cell/dropout/Const?
"dropout_lstm/lstm_cell/dropout/MulMul)dropout_lstm/lstm_cell/ones_like:output:0-dropout_lstm/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2$
"dropout_lstm/lstm_cell/dropout/Mul?
$dropout_lstm/lstm_cell/dropout/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$dropout_lstm/lstm_cell/dropout/Shape?
;dropout_lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform-dropout_lstm/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2??2=
;dropout_lstm/lstm_cell/dropout/random_uniform/RandomUniform?
-dropout_lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2/
-dropout_lstm/lstm_cell/dropout/GreaterEqual/y?
+dropout_lstm/lstm_cell/dropout/GreaterEqualGreaterEqualDdropout_lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:06dropout_lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2-
+dropout_lstm/lstm_cell/dropout/GreaterEqual?
#dropout_lstm/lstm_cell/dropout/CastCast/dropout_lstm/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2%
#dropout_lstm/lstm_cell/dropout/Cast?
$dropout_lstm/lstm_cell/dropout/Mul_1Mul&dropout_lstm/lstm_cell/dropout/Mul:z:0'dropout_lstm/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2&
$dropout_lstm/lstm_cell/dropout/Mul_1?
&dropout_lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_1/Const?
$dropout_lstm/lstm_cell/dropout_1/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2&
$dropout_lstm/lstm_cell/dropout_1/Mul?
&dropout_lstm/lstm_cell/dropout_1/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_1/Shape?
=dropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2??2?
=dropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_1/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2/
-dropout_lstm/lstm_cell/dropout_1/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_1/CastCast1dropout_lstm/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2'
%dropout_lstm/lstm_cell/dropout_1/Cast?
&dropout_lstm/lstm_cell/dropout_1/Mul_1Mul(dropout_lstm/lstm_cell/dropout_1/Mul:z:0)dropout_lstm/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2(
&dropout_lstm/lstm_cell/dropout_1/Mul_1?
&dropout_lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_2/Const?
$dropout_lstm/lstm_cell/dropout_2/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2&
$dropout_lstm/lstm_cell/dropout_2/Mul?
&dropout_lstm/lstm_cell/dropout_2/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_2/Shape?
=dropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_2/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2/
-dropout_lstm/lstm_cell/dropout_2/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_2/CastCast1dropout_lstm/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2'
%dropout_lstm/lstm_cell/dropout_2/Cast?
&dropout_lstm/lstm_cell/dropout_2/Mul_1Mul(dropout_lstm/lstm_cell/dropout_2/Mul:z:0)dropout_lstm/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2(
&dropout_lstm/lstm_cell/dropout_2/Mul_1?
&dropout_lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_3/Const?
$dropout_lstm/lstm_cell/dropout_3/MulMul)dropout_lstm/lstm_cell/ones_like:output:0/dropout_lstm/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2&
$dropout_lstm/lstm_cell/dropout_3/Mul?
&dropout_lstm/lstm_cell/dropout_3/ShapeShape)dropout_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_3/Shape?
=dropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_3/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2/
-dropout_lstm/lstm_cell/dropout_3/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_3/CastCast1dropout_lstm/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2'
%dropout_lstm/lstm_cell/dropout_3/Cast?
&dropout_lstm/lstm_cell/dropout_3/Mul_1Mul(dropout_lstm/lstm_cell/dropout_3/Mul:z:0)dropout_lstm/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2(
&dropout_lstm/lstm_cell/dropout_3/Mul_1?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
(dropout_lstm/lstm_cell/ones_like_1/ShapeShape/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_1/Shape?
(dropout_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_1/Const?
"dropout_lstm/lstm_cell/ones_like_1Fill1dropout_lstm/lstm_cell/ones_like_1/Shape:output:01dropout_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_1?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
(dropout_lstm/lstm_cell/ones_like_2/ShapeShape/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_2/Shape?
(dropout_lstm/lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_2/Const?
"dropout_lstm/lstm_cell/ones_like_2Fill1dropout_lstm/lstm_cell/ones_like_2/Shape:output:01dropout_lstm/lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_2?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
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
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
(dropout_lstm/lstm_cell/ones_like_3/ShapeShape/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_3/Shape?
(dropout_lstm/lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_3/Const?
"dropout_lstm/lstm_cell/ones_like_3Fill1dropout_lstm/lstm_cell/ones_like_3/Shape:output:01dropout_lstm/lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_3?
,dropout_lstm/lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_4/stack?
.dropout_lstm/lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_4/stack_1?
.dropout_lstm/lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_4/stack_2?
&dropout_lstm/lstm_cell/strided_slice_4StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_4/stack:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_4?
(dropout_lstm/lstm_cell/ones_like_4/ShapeShape/dropout_lstm/lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_4/Shape?
(dropout_lstm/lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_4/Const?
"dropout_lstm/lstm_cell/ones_like_4Fill1dropout_lstm/lstm_cell/ones_like_4/Shape:output:01dropout_lstm/lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_4?
"dropout_lstm/lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"dropout_lstm/lstm_cell/concat/axis?
dropout_lstm/lstm_cell/concatConcatV2(dropout_lstm/lstm_cell/dropout/Mul_1:z:0+dropout_lstm/lstm_cell/ones_like_1:output:0+dropout_lstm/lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/concat?
$dropout_lstm/lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_1/axis?
dropout_lstm/lstm_cell/concat_1ConcatV2*dropout_lstm/lstm_cell/dropout_1/Mul_1:z:0+dropout_lstm/lstm_cell/ones_like_2:output:0-dropout_lstm/lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_1?
$dropout_lstm/lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_2/axis?
dropout_lstm/lstm_cell/concat_2ConcatV2*dropout_lstm/lstm_cell/dropout_2/Mul_1:z:0+dropout_lstm/lstm_cell/ones_like_3:output:0-dropout_lstm/lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_2?
$dropout_lstm/lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_3/axis?
dropout_lstm/lstm_cell/concat_3ConcatV2*dropout_lstm/lstm_cell/dropout_3/Mul_1:z:0+dropout_lstm/lstm_cell/ones_like_4:output:0-dropout_lstm/lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_3?
(dropout_lstm/lstm_cell/ones_like_5/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_5/Shape?
(dropout_lstm/lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_5/Const?
"dropout_lstm/lstm_cell/ones_like_5Fill1dropout_lstm/lstm_cell/ones_like_5/Shape:output:01dropout_lstm/lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2$
"dropout_lstm/lstm_cell/ones_like_5?
&dropout_lstm/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_4/Const?
$dropout_lstm/lstm_cell/dropout_4/MulMul+dropout_lstm/lstm_cell/ones_like_5:output:0/dropout_lstm/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_4/Mul?
&dropout_lstm/lstm_cell/dropout_4/ShapeShape+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_4/Shape?
=dropout_lstm/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_4/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_4/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_4/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_4/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_4/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_4/CastCast1dropout_lstm/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_4/Cast?
&dropout_lstm/lstm_cell/dropout_4/Mul_1Mul(dropout_lstm/lstm_cell/dropout_4/Mul:z:0)dropout_lstm/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_4/Mul_1?
&dropout_lstm/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_5/Const?
$dropout_lstm/lstm_cell/dropout_5/MulMul+dropout_lstm/lstm_cell/ones_like_5:output:0/dropout_lstm/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_5/Mul?
&dropout_lstm/lstm_cell/dropout_5/ShapeShape+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_5/Shape?
=dropout_lstm/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2絋2?
=dropout_lstm/lstm_cell/dropout_5/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_5/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_5/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_5/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_5/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_5/CastCast1dropout_lstm/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_5/Cast?
&dropout_lstm/lstm_cell/dropout_5/Mul_1Mul(dropout_lstm/lstm_cell/dropout_5/Mul:z:0)dropout_lstm/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_5/Mul_1?
&dropout_lstm/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_6/Const?
$dropout_lstm/lstm_cell/dropout_6/MulMul+dropout_lstm/lstm_cell/ones_like_5:output:0/dropout_lstm/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_6/Mul?
&dropout_lstm/lstm_cell/dropout_6/ShapeShape+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_6/Shape?
=dropout_lstm/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??h2?
=dropout_lstm/lstm_cell/dropout_6/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_6/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_6/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_6/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_6/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_6/CastCast1dropout_lstm/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_6/Cast?
&dropout_lstm/lstm_cell/dropout_6/Mul_1Mul(dropout_lstm/lstm_cell/dropout_6/Mul:z:0)dropout_lstm/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_6/Mul_1?
&dropout_lstm/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&dropout_lstm/lstm_cell/dropout_7/Const?
$dropout_lstm/lstm_cell/dropout_7/MulMul+dropout_lstm/lstm_cell/ones_like_5:output:0/dropout_lstm/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2&
$dropout_lstm/lstm_cell/dropout_7/Mul?
&dropout_lstm/lstm_cell/dropout_7/ShapeShape+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2(
&dropout_lstm/lstm_cell/dropout_7/Shape?
=dropout_lstm/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform/dropout_lstm/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=dropout_lstm/lstm_cell/dropout_7/random_uniform/RandomUniform?
/dropout_lstm/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=21
/dropout_lstm/lstm_cell/dropout_7/GreaterEqual/y?
-dropout_lstm/lstm_cell/dropout_7/GreaterEqualGreaterEqualFdropout_lstm/lstm_cell/dropout_7/random_uniform/RandomUniform:output:08dropout_lstm/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-dropout_lstm/lstm_cell/dropout_7/GreaterEqual?
%dropout_lstm/lstm_cell/dropout_7/CastCast1dropout_lstm/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%dropout_lstm/lstm_cell/dropout_7/Cast?
&dropout_lstm/lstm_cell/dropout_7/Mul_1Mul(dropout_lstm/lstm_cell/dropout_7/Mul:z:0)dropout_lstm/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&dropout_lstm/lstm_cell/dropout_7/Mul_1?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:0&dropout_lstm/lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
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
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0*dropout_lstm/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
,dropout_lstm/lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,dropout_lstm/lstm_cell/strided_slice_5/stack?
.dropout_lstm/lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_5/stack_1?
.dropout_lstm/lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_5/stack_2?
&dropout_lstm/lstm_cell/strided_slice_5StridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:05dropout_lstm/lstm_cell/strided_slice_5/stack:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_5?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0/dropout_lstm/lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_6/stack?
.dropout_lstm/lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_6/stack_1?
.dropout_lstm/lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_6/stack_2?
&dropout_lstm/lstm_cell/strided_slice_6StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_6/stack:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_6?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_7/stack?
.dropout_lstm/lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_7/stack_1?
.dropout_lstm/lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_7/stack_2?
&dropout_lstm/lstm_cell/strided_slice_7StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_7/stack:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_7?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_8/stack?
.dropout_lstm/lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_8/stack_1?
.dropout_lstm/lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_8/stack_2?
&dropout_lstm/lstm_cell/strided_slice_8StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_8/stack:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_8?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_4346291*+
cond#R!
dropout_lstm_while_cond_4346290*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
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
valueB"    ?  2
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
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
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
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
IdentityIdentitystrided_slice:output:0^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_1:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::2(
dropout_lstm/whiledropout_lstm/while:U Q
,
_output_shapes
:??????????A
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
?
?
,__inference_sequential_layer_call_fn_4349476

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
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_43450092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_4349861

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4349437
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
B :?2
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
B :?2
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2	
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
 :??????????????????A2
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
valueB"????A   27
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
:?????????A*
shrink_axis_mask2
strided_slice_2?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlicestrided_slice_2:output:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3x
lstm_cell/ones_like_5/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4349259*
condR
while_cond_4349258*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????A:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????A
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
.__inference_dropout_lstm_layer_call_fn_4349452
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
I:???????????????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_43448062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????A:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????A
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
?
?
,__inference_sequential_layer_call_fn_4345016
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
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_43450092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
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
G__inference_sequential_layer_call_and_return_conditional_losses_4349495

inputs7
3mu_logstd_logmix_net_matmul_readvariableop_resource8
4mu_logstd_logmix_net_biasadd_readvariableop_resource
identity??
*mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp3mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*mu_logstd_logmix_net/MatMul/ReadVariableOp?
mu_logstd_logmix_net/MatMulMatMulinputs2mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mu_logstd_logmix_net/MatMul?
+mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp4mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
mu_logstd_logmix_net/BiasAddBiasAdd%mu_logstd_logmix_net/MatMul:product:03mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mu_logstd_logmix_net/BiasAddz
IdentityIdentity%mu_logstd_logmix_net/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_4344872
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_4344872___redundant_placeholder0/
+while_cond_4344872___redundant_placeholder1/
+while_cond_4344872___redundant_placeholder2/
+while_cond_4344872___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
G__inference_sequential_layer_call_and_return_conditional_losses_4344988
input_1 
mu_logstd_logmix_net_4344982 
mu_logstd_logmix_net_4344984
identity??,mu_logstd_logmix_net/StatefulPartitionedCall?
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinput_1mu_logstd_logmix_net_4344982mu_logstd_logmix_net_4344984*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_43449712.
,mu_logstd_logmix_net/StatefulPartitionedCall?
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0-^mu_logstd_logmix_net/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
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
(__inference_mdnrnn_layer_call_fn_4347803

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*;
_output_shapes)
':??????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_mdnrnn_layer_call_and_return_conditional_losses_43460252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????A
 
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
??
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4347769

inputs8
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1??dropout_lstm/while|
dropout_lstm/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:??????????A2
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
B :?2
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
B :?2
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
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2
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
:??????????A2
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
valueB"????A   2D
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
:?????????A*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
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
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice%dropout_lstm/strided_slice_2:output:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2&
$dropout_lstm/lstm_cell/strided_slice?
&dropout_lstm/lstm_cell/ones_like/ShapeShape-dropout_lstm/lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2"
 dropout_lstm/lstm_cell/ones_like?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
(dropout_lstm/lstm_cell/ones_like_1/ShapeShape/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_1/Shape?
(dropout_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_1/Const?
"dropout_lstm/lstm_cell/ones_like_1Fill1dropout_lstm/lstm_cell/ones_like_1/Shape:output:01dropout_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_1?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
(dropout_lstm/lstm_cell/ones_like_2/ShapeShape/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_2/Shape?
(dropout_lstm/lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_2/Const?
"dropout_lstm/lstm_cell/ones_like_2Fill1dropout_lstm/lstm_cell/ones_like_2/Shape:output:01dropout_lstm/lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_2?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
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
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
(dropout_lstm/lstm_cell/ones_like_3/ShapeShape/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_3/Shape?
(dropout_lstm/lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_3/Const?
"dropout_lstm/lstm_cell/ones_like_3Fill1dropout_lstm/lstm_cell/ones_like_3/Shape:output:01dropout_lstm/lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_3?
,dropout_lstm/lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_4/stack?
.dropout_lstm/lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_4/stack_1?
.dropout_lstm/lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_4/stack_2?
&dropout_lstm/lstm_cell/strided_slice_4StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_4/stack:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_4?
(dropout_lstm/lstm_cell/ones_like_4/ShapeShape/dropout_lstm/lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_4/Shape?
(dropout_lstm/lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_4/Const?
"dropout_lstm/lstm_cell/ones_like_4Fill1dropout_lstm/lstm_cell/ones_like_4/Shape:output:01dropout_lstm/lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_4?
"dropout_lstm/lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"dropout_lstm/lstm_cell/concat/axis?
dropout_lstm/lstm_cell/concatConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_1:output:0+dropout_lstm/lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/concat?
$dropout_lstm/lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_1/axis?
dropout_lstm/lstm_cell/concat_1ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_2:output:0-dropout_lstm/lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_1?
$dropout_lstm/lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_2/axis?
dropout_lstm/lstm_cell/concat_2ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_3:output:0-dropout_lstm/lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_2?
$dropout_lstm/lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_3/axis?
dropout_lstm/lstm_cell/concat_3ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_4:output:0-dropout_lstm/lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_3?
(dropout_lstm/lstm_cell/ones_like_5/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_5/Shape?
(dropout_lstm/lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_5/Const?
"dropout_lstm/lstm_cell/ones_like_5Fill1dropout_lstm/lstm_cell/ones_like_5/Shape:output:01dropout_lstm/lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2$
"dropout_lstm/lstm_cell/ones_like_5?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:0&dropout_lstm/lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
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
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
,dropout_lstm/lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,dropout_lstm/lstm_cell/strided_slice_5/stack?
.dropout_lstm/lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_5/stack_1?
.dropout_lstm/lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_5/stack_2?
&dropout_lstm/lstm_cell/strided_slice_5StridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:05dropout_lstm/lstm_cell/strided_slice_5/stack:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_5?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0/dropout_lstm/lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_6/stack?
.dropout_lstm/lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_6/stack_1?
.dropout_lstm/lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_6/stack_2?
&dropout_lstm/lstm_cell/strided_slice_6StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_6/stack:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_6?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_7/stack?
.dropout_lstm/lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_7/stack_1?
.dropout_lstm/lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_7/stack_2?
&dropout_lstm/lstm_cell/strided_slice_7StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_7/stack:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_7?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_8/stack?
.dropout_lstm/lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_8/stack_1?
.dropout_lstm/lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_8/stack_2?
&dropout_lstm/lstm_cell/strided_slice_8StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_8/stack:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_8?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_4347576*+
cond#R!
dropout_lstm_while_cond_4347575*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
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
valueB"    ?  2
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
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
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
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
IdentityIdentitystrided_slice:output:0^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_1:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::2(
dropout_lstm/whiledropout_lstm/while:T P
,
_output_shapes
:??????????A
 
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
?H
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4344944

inputs
lstm_cell_4344860
lstm_cell_4344862
lstm_cell_4344864
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
B :?2
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
B :?2
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2	
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
 :??????????????????A2
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
valueB"????A   27
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
:?????????A*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4344860lstm_cell_4344862lstm_cell_4344864*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_43444592#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4344860lstm_cell_4344862lstm_cell_4344864*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4344873*
condR
while_cond_4344872*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identitywhile:output:5"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????A:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????A
 
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
?
?
,__inference_sequential_layer_call_fn_4345034
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
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_43450272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: 
?w
?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_4349851

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2?{
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
:?????????@*

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
:?????????@2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????A2
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
:?????????A2

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
:?????????A2

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
:?????????A2

concat_3^
ones_like_5/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_5/Shapek
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_5/Const?
ones_like_5Fillones_like_5/Shape:output:0ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_5\
mulMulinputsconcat:output:0*
T0*'
_output_shapes
:?????????A2
mulb
mul_1Mulinputsconcat_1:output:0*
T0*'
_output_shapes
:?????????A2
mul_1b
mul_2Mulinputsconcat_2:output:0*
T0*'
_output_shapes
:?????????A2
mul_2b
mul_3Mulinputsconcat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

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
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3h
mul_4Mulstates_0ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
mul_4h
mul_5Mulstates_0ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
mul_5h
mul_6Mulstates_0ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
mul_6h
mul_7Mulstates_0ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_5v
MatMul_4MatMul	mul_4:z:0strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_1:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_6v
MatMul_5MatMul	mul_5:z:0strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_2:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_7v
MatMul_6MatMul	mul_6:z:0strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_3:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_8v
MatMul_7MatMul	mul_7:z:0strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????A:??????????:??????????::::O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
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
?
?
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_4344971

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
? 
?
while_body_4344735
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_4344759_0
lstm_cell_4344761_0
lstm_cell_4344763_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_4344759
lstm_cell_4344761
lstm_cell_4344763??!lstm_cell/StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_4344759_0lstm_cell_4344761_0lstm_cell_4344763_0*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_43443182#
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
:??????????2

Identity_4?

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"(
lstm_cell_4344759lstm_cell_4344759_0"(
lstm_cell_4344761lstm_cell_4344761_0"(
lstm_cell_4344763lstm_cell_4344763_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2F
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
:??????????:.*
(
_output_shapes
:??????????:
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
F__inference_lstm_cell_layer_call_and_return_conditional_losses_4344318

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2?{
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
:?????????@*

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
:?????????@2
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
:?????????@2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???2&
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
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
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
:?????????@2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2?ό2(
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
:?????????@2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
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
:?????????@2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
:?????????@2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
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
:?????????@2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
:?????????@2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????*
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
:?????????2
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
:?????????A2
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
:?????????A2

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
:?????????A2

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
:?????????A2

concat_3\
ones_like_5/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_5/Shapek
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_5/Const?
ones_like_5Fillones_like_5/Shape:output:0ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_5g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_4/Const?
dropout_4/MulMulones_like_5:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_5:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?? 2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_5/Const?
dropout_5/MulMulones_like_5:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_5:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ܶ?2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_6/Const?
dropout_6/MulMulones_like_5:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_5:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_7/Const?
dropout_7/MulMulones_like_5:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_5:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mul_1\
mulMulinputsconcat:output:0*
T0*'
_output_shapes
:?????????A2
mulb
mul_1Mulinputsconcat_1:output:0*
T0*'
_output_shapes
:?????????A2
mul_1b
mul_2Mulinputsconcat_2:output:0*
T0*'
_output_shapes
:?????????A2
mul_2b
mul_3Mulinputsconcat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

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
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3e
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4e
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5e
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6e
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_5v
MatMul_4MatMul	mul_4:z:0strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_1:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_6v
MatMul_5MatMul	mul_5:z:0strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_2:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_7v
MatMul_6MatMul	mul_6:z:0strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_3:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_8v
MatMul_7MatMul	mul_7:z:0strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:??????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????A:??????????:??????????::::O K
'
_output_shapes
:?????????A
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
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
??
?
while_body_4348026
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2֕22
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/dropout/Mul_1:z:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/dropout_1/Mul_1:z:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/dropout_2/Mul_1:z:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/dropout_3/Mul_1:z:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?˓22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?Ƙ22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
ـ
?
dropout_lstm_while_body_4346291#
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2˷?20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2ق222
0lstm_cell/dropout_1/random_uniform/RandomUniform?
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_1/GreaterEqual/y?
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2⦞22
0lstm_cell/dropout_2/random_uniform/RandomUniform?
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_2/GreaterEqual/y?
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/dropout/Mul_1:z:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/dropout_1/Mul_1:z:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/dropout_2/Mul_1:z:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/dropout_3/Mul_1:z:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ڝ22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??@22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
6__inference_mu_logstd_logmix_net_layer_call_fn_4349870

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
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_43449712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4348605

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
B :?2
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
B :?2
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2	
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
:??????????A2
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
valueB"????A   27
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
:?????????A*
shrink_axis_mask2
strided_slice_2?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlicestrided_slice_2:output:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlicestrided_slice_2:output:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3x
lstm_cell/ones_like_5/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4348427*
condR
while_cond_4348426*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
:???????????2

Identityo

Identity_1Identitywhile:output:4^while*
T0*(
_output_shapes
:??????????2

Identity_1o

Identity_2Identitywhile:output:5^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????A:::2
whilewhile:T P
,
_output_shapes
:??????????A
 
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
while_cond_4344734
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_4344734___redundant_placeholder0/
+while_cond_4344734___redundant_placeholder1/
+while_cond_4344734___redundant_placeholder2/
+while_cond_4344734___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
while_cond_4345661
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_4345661___redundant_placeholder0/
+while_cond_4345661___redundant_placeholder1/
+while_cond_4345661___redundant_placeholder2/
+while_cond_4345661___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
??
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4346901
input_18
4dropout_lstm_lstm_cell_split_readvariableop_resource:
6dropout_lstm_lstm_cell_split_1_readvariableop_resource2
.dropout_lstm_lstm_cell_readvariableop_resourceB
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resourceC
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource
identity

identity_1??dropout_lstm/while}
dropout_lstm/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:??????????A2
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
B :?2
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
B :?2
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
:??????????2
dropout_lstm/zeros{
dropout_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2
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
:??????????A2
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
valueB"????A   2D
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
:?????????A*
shrink_axis_mask2
dropout_lstm/strided_slice_2?
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
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice/stack_1?
,dropout_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,dropout_lstm/lstm_cell/strided_slice/stack_2?
$dropout_lstm/lstm_cell/strided_sliceStridedSlice%dropout_lstm/strided_slice_2:output:03dropout_lstm/lstm_cell/strided_slice/stack:output:05dropout_lstm/lstm_cell/strided_slice/stack_1:output:05dropout_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2&
$dropout_lstm/lstm_cell/strided_slice?
&dropout_lstm/lstm_cell/ones_like/ShapeShape-dropout_lstm/lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2"
 dropout_lstm/lstm_cell/ones_like?
,dropout_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_1/stack?
.dropout_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_1/stack_1?
.dropout_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_1/stack_2?
&dropout_lstm/lstm_cell/strided_slice_1StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_1/stack:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_1?
(dropout_lstm/lstm_cell/ones_like_1/ShapeShape/dropout_lstm/lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_1/Shape?
(dropout_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_1/Const?
"dropout_lstm/lstm_cell/ones_like_1Fill1dropout_lstm/lstm_cell/ones_like_1/Shape:output:01dropout_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_1?
,dropout_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_2/stack?
.dropout_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_2/stack_1?
.dropout_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_2/stack_2?
&dropout_lstm/lstm_cell/strided_slice_2StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_2/stack:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_2?
(dropout_lstm/lstm_cell/ones_like_2/ShapeShape/dropout_lstm/lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_2/Shape?
(dropout_lstm/lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_2/Const?
"dropout_lstm/lstm_cell/ones_like_2Fill1dropout_lstm/lstm_cell/ones_like_2/Shape:output:01dropout_lstm/lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_2?
,dropout_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
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
&dropout_lstm/lstm_cell/strided_slice_3StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_3/stack:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_3?
(dropout_lstm/lstm_cell/ones_like_3/ShapeShape/dropout_lstm/lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_3/Shape?
(dropout_lstm/lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_3/Const?
"dropout_lstm/lstm_cell/ones_like_3Fill1dropout_lstm/lstm_cell/ones_like_3/Shape:output:01dropout_lstm/lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_3?
,dropout_lstm/lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2.
,dropout_lstm/lstm_cell/strided_slice_4/stack?
.dropout_lstm/lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_4/stack_1?
.dropout_lstm/lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_4/stack_2?
&dropout_lstm/lstm_cell/strided_slice_4StridedSlice%dropout_lstm/strided_slice_2:output:05dropout_lstm/lstm_cell/strided_slice_4/stack:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_4?
(dropout_lstm/lstm_cell/ones_like_4/ShapeShape/dropout_lstm/lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_4/Shape?
(dropout_lstm/lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_4/Const?
"dropout_lstm/lstm_cell/ones_like_4Fill1dropout_lstm/lstm_cell/ones_like_4/Shape:output:01dropout_lstm/lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2$
"dropout_lstm/lstm_cell/ones_like_4?
"dropout_lstm/lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"dropout_lstm/lstm_cell/concat/axis?
dropout_lstm/lstm_cell/concatConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_1:output:0+dropout_lstm/lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/concat?
$dropout_lstm/lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_1/axis?
dropout_lstm/lstm_cell/concat_1ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_2:output:0-dropout_lstm/lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_1?
$dropout_lstm/lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_2/axis?
dropout_lstm/lstm_cell/concat_2ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_3:output:0-dropout_lstm/lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_2?
$dropout_lstm/lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$dropout_lstm/lstm_cell/concat_3/axis?
dropout_lstm/lstm_cell/concat_3ConcatV2)dropout_lstm/lstm_cell/ones_like:output:0+dropout_lstm/lstm_cell/ones_like_4:output:0-dropout_lstm/lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2!
dropout_lstm/lstm_cell/concat_3?
(dropout_lstm/lstm_cell/ones_like_5/ShapeShapedropout_lstm/zeros:output:0*
T0*
_output_shapes
:2*
(dropout_lstm/lstm_cell/ones_like_5/Shape?
(dropout_lstm/lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dropout_lstm/lstm_cell/ones_like_5/Const?
"dropout_lstm/lstm_cell/ones_like_5Fill1dropout_lstm/lstm_cell/ones_like_5/Shape:output:01dropout_lstm/lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2$
"dropout_lstm/lstm_cell/ones_like_5?
dropout_lstm/lstm_cell/mulMul%dropout_lstm/strided_slice_2:output:0&dropout_lstm/lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul?
dropout_lstm/lstm_cell/mul_1Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_1?
dropout_lstm/lstm_cell/mul_2Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
dropout_lstm/lstm_cell/mul_2?
dropout_lstm/lstm_cell/mul_3Mul%dropout_lstm/strided_slice_2:output:0(dropout_lstm/lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02-
+dropout_lstm/lstm_cell/split/ReadVariableOp?
dropout_lstm/lstm_cell/splitSplit/dropout_lstm/lstm_cell/split/split_dim:output:03dropout_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
dropout_lstm/lstm_cell/split?
dropout_lstm/lstm_cell/MatMulMatMuldropout_lstm/lstm_cell/mul:z:0%dropout_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/MatMul?
dropout_lstm/lstm_cell/MatMul_1MatMul dropout_lstm/lstm_cell/mul_1:z:0%dropout_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_1?
dropout_lstm/lstm_cell/MatMul_2MatMul dropout_lstm/lstm_cell/mul_2:z:0%dropout_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_2?
dropout_lstm/lstm_cell/MatMul_3MatMul dropout_lstm/lstm_cell/mul_3:z:0%dropout_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2!
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
:?*
dtype02/
-dropout_lstm/lstm_cell/split_1/ReadVariableOp?
dropout_lstm/lstm_cell/split_1Split1dropout_lstm/lstm_cell/split_1/split_dim:output:05dropout_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2 
dropout_lstm/lstm_cell/split_1?
dropout_lstm/lstm_cell/BiasAddBiasAdd'dropout_lstm/lstm_cell/MatMul:product:0'dropout_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/BiasAdd?
 dropout_lstm/lstm_cell/BiasAdd_1BiasAdd)dropout_lstm/lstm_cell/MatMul_1:product:0'dropout_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_1?
 dropout_lstm/lstm_cell/BiasAdd_2BiasAdd)dropout_lstm/lstm_cell/MatMul_2:product:0'dropout_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_2?
 dropout_lstm/lstm_cell/BiasAdd_3BiasAdd)dropout_lstm/lstm_cell/MatMul_3:product:0'dropout_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/BiasAdd_3?
dropout_lstm/lstm_cell/mul_4Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_4?
dropout_lstm/lstm_cell/mul_5Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_5?
dropout_lstm/lstm_cell/mul_6Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_6?
dropout_lstm/lstm_cell/mul_7Muldropout_lstm/zeros:output:0+dropout_lstm/lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_7?
%dropout_lstm/lstm_cell/ReadVariableOpReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%dropout_lstm/lstm_cell/ReadVariableOp?
,dropout_lstm/lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,dropout_lstm/lstm_cell/strided_slice_5/stack?
.dropout_lstm/lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_5/stack_1?
.dropout_lstm/lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_5/stack_2?
&dropout_lstm/lstm_cell/strided_slice_5StridedSlice-dropout_lstm/lstm_cell/ReadVariableOp:value:05dropout_lstm/lstm_cell/strided_slice_5/stack:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_5?
dropout_lstm/lstm_cell/MatMul_4MatMul dropout_lstm/lstm_cell/mul_4:z:0/dropout_lstm/lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_4?
dropout_lstm/lstm_cell/addAddV2'dropout_lstm/lstm_cell/BiasAdd:output:0)dropout_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add?
dropout_lstm/lstm_cell/SigmoidSigmoiddropout_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2 
dropout_lstm/lstm_cell/Sigmoid?
'dropout_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_1?
,dropout_lstm/lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_6/stack?
.dropout_lstm/lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_6/stack_1?
.dropout_lstm/lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_6/stack_2?
&dropout_lstm/lstm_cell/strided_slice_6StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_1:value:05dropout_lstm/lstm_cell/strided_slice_6/stack:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_6?
dropout_lstm/lstm_cell/MatMul_5MatMul dropout_lstm/lstm_cell/mul_5:z:0/dropout_lstm/lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_5?
dropout_lstm/lstm_cell/add_1AddV2)dropout_lstm/lstm_cell/BiasAdd_1:output:0)dropout_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_1?
 dropout_lstm/lstm_cell/Sigmoid_1Sigmoid dropout_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_1?
dropout_lstm/lstm_cell/mul_8Mul$dropout_lstm/lstm_cell/Sigmoid_1:y:0dropout_lstm/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_8?
'dropout_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_2?
,dropout_lstm/lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_7/stack?
.dropout_lstm/lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.dropout_lstm/lstm_cell/strided_slice_7/stack_1?
.dropout_lstm/lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_7/stack_2?
&dropout_lstm/lstm_cell/strided_slice_7StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_2:value:05dropout_lstm/lstm_cell/strided_slice_7/stack:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_7?
dropout_lstm/lstm_cell/MatMul_6MatMul dropout_lstm/lstm_cell/mul_6:z:0/dropout_lstm/lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_6?
dropout_lstm/lstm_cell/add_2AddV2)dropout_lstm/lstm_cell/BiasAdd_2:output:0)dropout_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_2?
dropout_lstm/lstm_cell/TanhTanh dropout_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh?
dropout_lstm/lstm_cell/mul_9Mul"dropout_lstm/lstm_cell/Sigmoid:y:0dropout_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_9?
dropout_lstm/lstm_cell/add_3AddV2 dropout_lstm/lstm_cell/mul_8:z:0 dropout_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_3?
'dropout_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp.dropout_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'dropout_lstm/lstm_cell/ReadVariableOp_3?
,dropout_lstm/lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,dropout_lstm/lstm_cell/strided_slice_8/stack?
.dropout_lstm/lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.dropout_lstm/lstm_cell/strided_slice_8/stack_1?
.dropout_lstm/lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.dropout_lstm/lstm_cell/strided_slice_8/stack_2?
&dropout_lstm/lstm_cell/strided_slice_8StridedSlice/dropout_lstm/lstm_cell/ReadVariableOp_3:value:05dropout_lstm/lstm_cell/strided_slice_8/stack:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_1:output:07dropout_lstm/lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&dropout_lstm/lstm_cell/strided_slice_8?
dropout_lstm/lstm_cell/MatMul_7MatMul dropout_lstm/lstm_cell/mul_7:z:0/dropout_lstm/lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2!
dropout_lstm/lstm_cell/MatMul_7?
dropout_lstm/lstm_cell/add_4AddV2)dropout_lstm/lstm_cell/BiasAdd_3:output:0)dropout_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/add_4?
 dropout_lstm/lstm_cell/Sigmoid_2Sigmoid dropout_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2"
 dropout_lstm/lstm_cell/Sigmoid_2?
dropout_lstm/lstm_cell/Tanh_1Tanh dropout_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/Tanh_1?
dropout_lstm/lstm_cell/mul_10Mul$dropout_lstm/lstm_cell/Sigmoid_2:y:0!dropout_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
dropout_lstm/lstm_cell/mul_10?
*dropout_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
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
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
dropout_lstm_while_body_4346708*+
cond#R!
dropout_lstm_while_cond_4346707*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
dropout_lstm/while?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=dropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape?
/dropout_lstm/TensorArrayV2Stack/TensorListStackTensorListStackdropout_lstm/while:output:3Fdropout_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
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
:??????????*
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
:???????????2
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
valueB"????   2
Reshape/shape?
ReshapeReshapedropout_lstm/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp?
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential/mu_logstd_logmix_net/MatMul?
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp?
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
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
valueB"    ?  2
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
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
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
strided_slice_1StridedSlice0sequential/mu_logstd_logmix_net/BiasAdd:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
IdentityIdentitystrided_slice:output:0^dropout_lstm/while*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_1:output:0^dropout_lstm/while*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::2(
dropout_lstm/whiledropout_lstm/while:U Q
,
_output_shapes
:??????????A
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
G__inference_sequential_layer_call_and_return_conditional_losses_4344997
input_1 
mu_logstd_logmix_net_4344991 
mu_logstd_logmix_net_4344993
identity??,mu_logstd_logmix_net/StatefulPartitionedCall?
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinput_1mu_logstd_logmix_net_4344991mu_logstd_logmix_net_4344993*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*Z
fURS
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_43449712.
,mu_logstd_logmix_net/StatefulPartitionedCall?
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0-^mu_logstd_logmix_net/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
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
.__inference_dropout_lstm_layer_call_fn_4348620

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
A:???????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_43455032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????A:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????A
 
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
while_cond_4345260
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_4345260___redundant_placeholder0/
+while_cond_4345260___redundant_placeholder1/
+while_cond_4345260___redundant_placeholder2/
+while_cond_4345260___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?H
?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4344806

inputs
lstm_cell_4344722
lstm_cell_4344724
lstm_cell_4344726
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
B :?2
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
B :?2
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
:??????????2	
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
 :??????????????????A2
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
valueB"????A   27
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
:?????????A*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4344722lstm_cell_4344724lstm_cell_4344726*
Tin

2*
Tout
2*P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_43443182#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4344722lstm_cell_4344724lstm_cell_4344726*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4344735*
condR
while_cond_4344734*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity?

Identity_1Identitywhile:output:4"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identitywhile:output:5"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:??????????????????A:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????A
 
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
??
?
while_body_4349259
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?
dropout_lstm_while_body_4347576#
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?
dropout_lstm_while_cond_4346707#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_4346707___redundant_placeholder0<
8dropout_lstm_while_cond_4346707___redundant_placeholder1<
8dropout_lstm_while_cond_4346707___redundant_placeholder2<
8dropout_lstm_while_cond_4346707___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
while_cond_4348426
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_4348426___redundant_placeholder0/
+while_cond_4348426___redundant_placeholder1/
+while_cond_4348426___redundant_placeholder2/
+while_cond_4348426___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?
while_body_4345662
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?
dropout_lstm_while_body_4346708#
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_dropout_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_like?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/ones_like:output:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
(__inference_mdnrnn_layer_call_fn_4346935
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*;
_output_shapes)
':??????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_mdnrnn_layer_call_and_return_conditional_losses_43460252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????A
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
?
while_body_4345261
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
valueB"????A   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????A*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
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
valueB"    ????2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice*TensorArrayV2Read/TensorListGetItem:item:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*

begin_mask*
ellipsis_mask2
lstm_cell/strided_slice?
lstm_cell/ones_like/ShapeShape lstm_cell/strided_slice:output:0*
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
T0*'
_output_shapes
:?????????@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout/Const?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape?
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???20
.lstm_cell/dropout/random_uniform/RandomUniform?
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 lstm_cell/dropout/GreaterEqual/y?
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2 
lstm_cell/dropout/GreaterEqual?
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout/Cast?
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul?
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape?
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_1/GreaterEqual?
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Cast?
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul?
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape?
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
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
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_2/GreaterEqual?
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Cast?
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul?
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape?
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_3/random_uniform/RandomUniform?
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_3/GreaterEqual/y?
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2"
 lstm_cell/dropout_3/GreaterEqual?
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Cast?
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@2
lstm_cell/dropout_3/Mul_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/ones_like_1/ShapeShape"lstm_cell/strided_slice_1:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_1/Const?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_1?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/ones_like_2/ShapeShape"lstm_cell/strided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_2/Shape
lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_2/Const?
lstm_cell/ones_like_2Fill$lstm_cell/ones_like_2/Shape:output:0$lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_2?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
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
lstm_cell/strided_slice_3StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/ones_like_3/ShapeShape"lstm_cell/strided_slice_3:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_3/Shape
lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_3/Const?
lstm_cell/ones_like_3Fill$lstm_cell/ones_like_3/Shape:output:0$lstm_cell/ones_like_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_3?
lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2!
lstm_cell/strided_slice_4/stack?
!lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_4/stack_1?
!lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_4/stack_2?
lstm_cell/strided_slice_4StridedSlice*TensorArrayV2Read/TensorListGetItem:item:0(lstm_cell/strided_slice_4/stack:output:0*lstm_cell/strided_slice_4/stack_1:output:0*lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
end_mask2
lstm_cell/strided_slice_4?
lstm_cell/ones_like_4/ShapeShape"lstm_cell/strided_slice_4:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_4/Shape
lstm_cell/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_4/Const?
lstm_cell/ones_like_4Fill$lstm_cell/ones_like_4/Shape:output:0$lstm_cell/ones_like_4/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/ones_like_4y
lstm_cell/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat/axis?
lstm_cell/concatConcatV2lstm_cell/dropout/Mul_1:z:0lstm_cell/ones_like_1:output:0lstm_cell/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat}
lstm_cell/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_1/axis?
lstm_cell/concat_1ConcatV2lstm_cell/dropout_1/Mul_1:z:0lstm_cell/ones_like_2:output:0 lstm_cell/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_1}
lstm_cell/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_2/axis?
lstm_cell/concat_2ConcatV2lstm_cell/dropout_2/Mul_1:z:0lstm_cell/ones_like_3:output:0 lstm_cell/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_2}
lstm_cell/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_cell/concat_3/axis?
lstm_cell/concat_3ConcatV2lstm_cell/dropout_3/Mul_1:z:0lstm_cell/ones_like_4:output:0 lstm_cell/concat_3/axis:output:0*
N*
T0*'
_output_shapes
:?????????A2
lstm_cell/concat_3w
lstm_cell/ones_like_5/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_5/Shape
lstm_cell/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell/ones_like_5/Const?
lstm_cell/ones_like_5Fill$lstm_cell/ones_like_5/Shape:output:0$lstm_cell/ones_like_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/ones_like_5{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_4/Const?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul?
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shape?
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??z22
0lstm_cell/dropout_4/random_uniform/RandomUniform?
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_4/GreaterEqual/y?
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_4/GreaterEqual?
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Cast?
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_5/Const?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul?
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shape?
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??-22
0lstm_cell/dropout_5/random_uniform/RandomUniform?
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_5/GreaterEqual/y?
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_5/GreaterEqual?
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Cast?
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_6/Const?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul?
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shape?
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_6/random_uniform/RandomUniform?
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_6/GreaterEqual/y?
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_6/GreaterEqual?
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Cast?
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell/dropout_7/Const?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_5:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul?
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_5:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shape?
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???22
0lstm_cell/dropout_7/random_uniform/RandomUniform?
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2$
"lstm_cell/dropout_7/GreaterEqual/y?
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_cell/dropout_7/GreaterEqual?
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Cast?
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/dropout_7/Mul_1?
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul?
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_1:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_1?
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_2:output:0*
T0*'
_output_shapes
:?????????A2
lstm_cell/mul_2?
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/concat_3:output:0*
T0*'
_output_shapes
:?????????A2
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
:	A?*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	A?:	A?:	A?:	A?*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
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
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_4?
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_6?
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_7?
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_5/stack?
!lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_5/stack_1?
!lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_5/stack_2?
lstm_cell/strided_slice_5StridedSlice lstm_cell/ReadVariableOp:value:0(lstm_cell/strided_slice_5/stack:output:0*lstm_cell/strided_slice_5/stack_1:output:0*lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_5?
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0"lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_6/stack?
!lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_6/stack_1?
!lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_6/stack_2?
lstm_cell/strided_slice_6StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_6/stack:output:0*lstm_cell/strided_slice_6/stack_1:output:0*lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_6?
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_8?
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_7/stack?
!lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_7/stack_1?
!lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_7/stack_2?
lstm_cell/strided_slice_7StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_7/stack:output:0*lstm_cell/strided_slice_7/stack_1:output:0*lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_7?
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh?
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_9?
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_3?
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_8/stack?
!lstm_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_8/stack_1?
!lstm_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_8/stack_2?
lstm_cell/strided_slice_8StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_8/stack:output:0*lstm_cell/strided_slice_8/stack_1:output:0*lstm_cell/strided_slice_8/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_8?
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Tanh_1?
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
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
:??????????2

Identity_4l

Identity_5Identitylstm_cell/add_3:z:0*
T0*(
_output_shapes
:??????????2

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
@: : : : :??????????:??????????: : :::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
%__inference_signature_wrapper_4346067
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*;
_output_shapes)
':??????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*+
f&R$
"__inference__wrapped_model_43440852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????A
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
(__inference_mdnrnn_layer_call_fn_4347786

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*;
_output_shapes)
':??????????:?????????*'
_read_only_resource_inputs	
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_mdnrnn_layer_call_and_return_conditional_losses_43459782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????A
 
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
?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4346025

inputs
dropout_lstm_4345999
dropout_lstm_4346001
dropout_lstm_4346003
sequential_4346010
sequential_4346012
identity

identity_1??$dropout_lstm/StatefulPartitionedCall?"sequential/StatefulPartitionedCall|
dropout_lstm/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:??????????A2
dropout_lstm/Cast?
$dropout_lstm/StatefulPartitionedCallStatefulPartitionedCalldropout_lstm/Cast:y:0dropout_lstm_4345999dropout_lstm_4346001dropout_lstm_4346003*
Tin
2*
Tout
2*U
_output_shapesC
A:???????????:??????????:??????????*%
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_43458402&
$dropout_lstm/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shape?
ReshapeReshape-dropout_lstm/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
"sequential/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0sequential_4346010sequential_4346012*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_43450272$
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
valueB"    ?  2
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
:??????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
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
strided_slice_1StridedSlice+sequential/StatefulPartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
IdentityIdentitystrided_slice:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identitystrided_slice_1:output:0%^dropout_lstm/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:??????????A:::::2L
$dropout_lstm/StatefulPartitionedCall$dropout_lstm/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
,
_output_shapes
:??????????A
 
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
dropout_lstm_while_cond_4347158#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_4347158___redundant_placeholder0<
8dropout_lstm_while_cond_4347158___redundant_placeholder1<
8dropout_lstm_while_cond_4347158___redundant_placeholder2<
8dropout_lstm_while_cond_4347158___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
?
dropout_lstm_while_cond_4346290#
dropout_lstm_while_loop_counter)
%dropout_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!less_dropout_lstm_strided_slice_1<
8dropout_lstm_while_cond_4346290___redundant_placeholder0<
8dropout_lstm_while_cond_4346290___redundant_placeholder1<
8dropout_lstm_while_cond_4346290___redundant_placeholder2<
8dropout_lstm_while_cond_4346290___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 
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
:??????????:.*
(
_output_shapes
:??????????:
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
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????A8
MDN1
StatefulPartitionedCall:0??????????5
d0
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?
	optimizer
loss_fn
inference_base
out_net
loss

signatures
regularization_losses
trainable_variables
		keras_api

	variables
[__call__
*\&call_and_return_all_conditional_losses
]_default_save_signature"?
_tf_keras_model?{"keras_version": "2.3.0-tf", "model_config": {"class_name": "MDNRNN"}, "batch_input_shape": null, "expects_training_arg": true, "backend": "tensorflow", "is_graph_network": false, "dtype": "float32", "class_name": "MDNRNN", "name": "mdnrnn", "config": {"layer was saved without config": true}, "training_config": {"loss": {"MDN": "z_loss_func", "d": "d_loss_func"}, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"decay": 0.0, "epsilon": 1e-07, "amsgrad": false, "learning_rate": 0.0006802849820815027, "beta_2": 0.9990000128746033, "beta_1": 0.8999999761581421, "clipvalue": 1.0, "name": "Adam"}}, "weighted_metrics": null, "loss_weights": null, "metrics": null}, "trainable": true}
?
iter

beta_1

beta_2
	decay
learning_ratemQmRmSmTmUvVvWvXvYvZ"
	optimizer
 "
trackable_dict_wrapper
?
cell

state_spec
regularization_losses
trainable_variables
	keras_api
	variables
^__call__
*_&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"trainable": true, "stateful": false, "dtype": "float32", "class_name": "DropoutLSTM", "name": "dropout_lstm", "config": {"activation": "tanh", "recurrent_dropout": 0.05, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_constraint": null, "return_state": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "stateful": false, "recurrent_activation": "sigmoid", "return_sequences": true, "bias_regularizer": null, "time_major": false, "dropout": 0.05, "implementation": 1, "trainable": true, "recurrent_constraint": null, "kernel_regularizer": null, "unit_forget_bias": true, "unroll": false, "kernel_constraint": null, "go_backwards": false, "dtype": "float32", "activity_regularizer": null, "name": "dropout_lstm", "units": 512, "use_bias": true, "recurrent_regularizer": null}, "batch_input_shape": null, "expects_training_arg": true, "build_input_shape": {"items": [100, 500, 65], "class_name": "TensorShape"}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "min_ndim": null, "axes": {}, "max_ndim": null, "shape": {"items": [null, null, 65], "class_name": "__tuple__"}, "ndim": 3}}]}
?
layer_with_weights-0
layer-0
regularization_losses
trainable_variables
	keras_api
	variables
`__call__
*a&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"keras_version": "2.3.0-tf", "model_config": {"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {"activation": "linear", "kernel_regularizer": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_constraint": null, "batch_input_shape": {"items": [null, 512], "class_name": "__tuple__"}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "dtype": "float32", "activity_regularizer": null, "name": "mu_logstd_logmix_net", "bias_regularizer": null, "units": 961, "use_bias": true, "trainable": true}}], "build_input_shape": {"items": [null, 512], "class_name": "TensorShape"}, "name": "sequential"}}, "batch_input_shape": null, "expects_training_arg": true, "build_input_shape": {"items": [null, 512], "class_name": "TensorShape"}, "backend": "tensorflow", "is_graph_network": true, "dtype": "float32", "class_name": "Sequential", "name": "sequential", "config": {"build_input_shape": {"items": [null, 512], "class_name": "TensorShape"}, "layers": [{"class_name": "Dense", "config": {"activation": "linear", "kernel_regularizer": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_constraint": null, "batch_input_shape": {"items": [null, 512], "class_name": "__tuple__"}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "dtype": "float32", "activity_regularizer": null, "name": "mu_logstd_logmix_net", "bias_regularizer": null, "units": 961, "use_bias": true, "trainable": true}}], "name": "sequential"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "min_ndim": 2, "axes": {"-1": 512}, "max_ndim": null, "shape": null, "ndim": null}}, "trainable": true}
 "
trackable_dict_wrapper
,
bserving_default"
signature_map
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
 non_trainable_variables
!layer_regularization_losses
regularization_losses

"layers
#layer_metrics
trainable_variables

	variables
$metrics
[__call__
&\"call_and_return_conditional_losses
*\&call_and_return_all_conditional_losses
]_default_save_signature"
_generic_user_object
C
0
1
2
3
4"
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
%regularization_losses
&trainable_variables
'	keras_api
(	variables
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"stateful": false, "dtype": "float32", "class_name": "LSTMCell", "name": "lstm_cell", "config": {"recurrent_regularizer": null, "activation": "tanh", "recurrent_activation": "sigmoid", "kernel_regularizer": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_constraint": null, "bias_constraint": null, "kernel_constraint": null, "units": 512, "bias_regularizer": null, "dtype": "float32", "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "unit_forget_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "recurrent_dropout": 0.05, "name": "lstm_cell", "dropout": 0.05, "use_bias": true, "implementation": 1, "trainable": true}, "batch_input_shape": null, "expects_training_arg": true, "trainable": true}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
)non_trainable_variables
*layer_regularization_losses

+states
regularization_losses

,layers
-layer_metrics
trainable_variables
	variables
.metrics
^__call__
&_"call_and_return_conditional_losses
*_&call_and_return_all_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
?

kernel
bias
/regularization_losses
0trainable_variables
1	keras_api
2	variables
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"trainable": true, "stateful": false, "dtype": "float32", "class_name": "Dense", "name": "mu_logstd_logmix_net", "config": {"activation": "linear", "kernel_regularizer": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_constraint": null, "bias_regularizer": null, "kernel_constraint": null, "dtype": "float32", "activity_regularizer": null, "name": "mu_logstd_logmix_net", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 961, "use_bias": true, "trainable": true}, "batch_input_shape": null, "expects_training_arg": false, "build_input_shape": {"items": [null, 512], "class_name": "TensorShape"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "min_ndim": 2, "axes": {"-1": 512}, "max_ndim": null, "shape": null, "ndim": null}}}
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
3non_trainable_variables
4layer_regularization_losses
regularization_losses

5layers
6layer_metrics
trainable_variables
	variables
7metrics
`__call__
&a"call_and_return_conditional_losses
*a&call_and_return_all_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
0:.	A?2dropout_lstm/lstm_cell/kernel
;:9
??2'dropout_lstm/lstm_cell/recurrent_kernel
*:(?2dropout_lstm/lstm_cell/bias
/:-
??2mu_logstd_logmix_net/kernel
(:&?2mu_logstd_logmix_net/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
5
80
91
:2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
;non_trainable_variables
<layer_regularization_losses
%regularization_losses

=layers
>layer_metrics
&trainable_variables
(	variables
?metrics
c__call__
&d"call_and_return_conditional_losses
*d&call_and_return_all_conditional_losses"
_generic_user_object
5
0
1
2"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
@non_trainable_variables
Alayer_regularization_losses
/regularization_losses

Blayers
Clayer_metrics
0trainable_variables
2	variables
Dmetrics
e__call__
&f"call_and_return_conditional_losses
*f&call_and_return_all_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	Etotal
	Fcount
G	variables
H	keras_api"?
_tf_keras_metricj{"dtype": "float32", "class_name": "Mean", "name": "loss", "config": {"dtype": "float32", "name": "loss"}}
?
	Itotal
	Jcount
K	variables
L	keras_api"?
_tf_keras_metricr{"dtype": "float32", "class_name": "Mean", "name": "MDN_loss", "config": {"dtype": "float32", "name": "MDN_loss"}}
?
	Mtotal
	Ncount
O	variables
P	keras_api"?
_tf_keras_metricn{"dtype": "float32", "class_name": "Mean", "name": "d_loss", "config": {"dtype": "float32", "name": "d_loss"}}
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
E0
F1"
trackable_list_wrapper
-
G	variables"
_generic_user_object
:  (2total
:  (2count
.
I0
J1"
trackable_list_wrapper
-
K	variables"
_generic_user_object
:  (2total
:  (2count
.
M0
N1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
5:3	A?2$Adam/dropout_lstm/lstm_cell/kernel/m
@:>
??2.Adam/dropout_lstm/lstm_cell/recurrent_kernel/m
/:-?2"Adam/dropout_lstm/lstm_cell/bias/m
4:2
??2"Adam/mu_logstd_logmix_net/kernel/m
-:+?2 Adam/mu_logstd_logmix_net/bias/m
5:3	A?2$Adam/dropout_lstm/lstm_cell/kernel/v
@:>
??2.Adam/dropout_lstm/lstm_cell/recurrent_kernel/v
/:-?2"Adam/dropout_lstm/lstm_cell/bias/v
4:2
??2"Adam/mu_logstd_logmix_net/kernel/v
-:+?2 Adam/mu_logstd_logmix_net/bias/v
?2?
(__inference_mdnrnn_layer_call_fn_4346935
(__inference_mdnrnn_layer_call_fn_4346918
(__inference_mdnrnn_layer_call_fn_4347803
(__inference_mdnrnn_layer_call_fn_4347786?
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
?2?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4346901
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4346548
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4347416
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4347769?
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
"__inference__wrapped_model_4344085?
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
input_1??????????A
?2?
.__inference_dropout_lstm_layer_call_fn_4349467
.__inference_dropout_lstm_layer_call_fn_4348635
.__inference_dropout_lstm_layer_call_fn_4348620
.__inference_dropout_lstm_layer_call_fn_4349452?
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
?2?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4349437
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4348268
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4348605
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4349100?
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
,__inference_sequential_layer_call_fn_4349476
,__inference_sequential_layer_call_fn_4345034
,__inference_sequential_layer_call_fn_4349485
,__inference_sequential_layer_call_fn_4345016?
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
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_4349495
G__inference_sequential_layer_call_and_return_conditional_losses_4349505
G__inference_sequential_layer_call_and_return_conditional_losses_4344997
G__inference_sequential_layer_call_and_return_conditional_losses_4344988?
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
%__inference_signature_wrapper_4346067input_1
?2?
+__inference_lstm_cell_layer_call_fn_4349539
+__inference_lstm_cell_layer_call_fn_4349522?
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
F__inference_lstm_cell_layer_call_and_return_conditional_losses_4349851
F__inference_lstm_cell_layer_call_and_return_conditional_losses_4349727?
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
6__inference_mu_logstd_logmix_net_layer_call_fn_4349870?
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
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_4349861?
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
"__inference__wrapped_model_4344085?5?2
+?(
&?#
input_1??????????A
? "L?I
%
MDN?
MDN??????????
 
d?
d??????????
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4348268?@?=
6?3
%?"
inputs??????????A

 
p

 
? "r?o
h?e
#? 
0/0???????????
?
0/1??????????
?
0/2??????????
? ?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4348605?@?=
6?3
%?"
inputs??????????A

 
p 

 
? "r?o
h?e
#? 
0/0???????????
?
0/1??????????
?
0/2??????????
? ?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4349100?O?L
E?B
4?1
/?,
inputs/0??????????????????A

 
p

 
? "z?w
p?m
+?(
0/0???????????????????
?
0/1??????????
?
0/2??????????
? ?
I__inference_dropout_lstm_layer_call_and_return_conditional_losses_4349437?O?L
E?B
4?1
/?,
inputs/0??????????????????A

 
p 

 
? "z?w
p?m
+?(
0/0???????????????????
?
0/1??????????
?
0/2??????????
? ?
.__inference_dropout_lstm_layer_call_fn_4348620?@?=
6?3
%?"
inputs??????????A

 
p

 
? "b?_
!?
0???????????
?
1??????????
?
2???????????
.__inference_dropout_lstm_layer_call_fn_4348635?@?=
6?3
%?"
inputs??????????A

 
p 

 
? "b?_
!?
0???????????
?
1??????????
?
2???????????
.__inference_dropout_lstm_layer_call_fn_4349452?O?L
E?B
4?1
/?,
inputs/0??????????????????A

 
p

 
? "j?g
)?&
0???????????????????
?
1??????????
?
2???????????
.__inference_dropout_lstm_layer_call_fn_4349467?O?L
E?B
4?1
/?,
inputs/0??????????????????A

 
p 

 
? "j?g
)?&
0???????????????????
?
1??????????
?
2???????????
F__inference_lstm_cell_layer_call_and_return_conditional_losses_4349727???
x?u
 ?
inputs?????????A
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
F__inference_lstm_cell_layer_call_and_return_conditional_losses_4349851???
x?u
 ?
inputs?????????A
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
+__inference_lstm_cell_layer_call_fn_4349522???
x?u
 ?
inputs?????????A
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
+__inference_lstm_cell_layer_call_fn_4349539???
x?u
 ?
inputs?????????A
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4346548?9?6
/?,
&?#
input_1??????????A
p
? "Z?W
P?M
'
MDN ?
0/MDN??????????
"
d?
0/d?????????
? ?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4346901?9?6
/?,
&?#
input_1??????????A
p 
? "Z?W
P?M
'
MDN ?
0/MDN??????????
"
d?
0/d?????????
? ?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4347416?8?5
.?+
%?"
inputs??????????A
p
? "Z?W
P?M
'
MDN ?
0/MDN??????????
"
d?
0/d?????????
? ?
C__inference_mdnrnn_layer_call_and_return_conditional_losses_4347769?8?5
.?+
%?"
inputs??????????A
p 
? "Z?W
P?M
'
MDN ?
0/MDN??????????
"
d?
0/d?????????
? ?
(__inference_mdnrnn_layer_call_fn_4346918?9?6
/?,
&?#
input_1??????????A
p
? "L?I
%
MDN?
MDN??????????
 
d?
d??????????
(__inference_mdnrnn_layer_call_fn_4346935?9?6
/?,
&?#
input_1??????????A
p 
? "L?I
%
MDN?
MDN??????????
 
d?
d??????????
(__inference_mdnrnn_layer_call_fn_4347786?8?5
.?+
%?"
inputs??????????A
p
? "L?I
%
MDN?
MDN??????????
 
d?
d??????????
(__inference_mdnrnn_layer_call_fn_4347803?8?5
.?+
%?"
inputs??????????A
p 
? "L?I
%
MDN?
MDN??????????
 
d?
d??????????
Q__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_4349861^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
6__inference_mu_logstd_logmix_net_layer_call_fn_4349870Q0?-
&?#
!?
inputs??????????
? "????????????
G__inference_sequential_layer_call_and_return_conditional_losses_4344988g9?6
/?,
"?
input_1??????????
p

 
? "&?#
?
0??????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_4344997g9?6
/?,
"?
input_1??????????
p 

 
? "&?#
?
0??????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_4349495f8?5
.?+
!?
inputs??????????
p

 
? "&?#
?
0??????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_4349505f8?5
.?+
!?
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
,__inference_sequential_layer_call_fn_4345016Z9?6
/?,
"?
input_1??????????
p

 
? "????????????
,__inference_sequential_layer_call_fn_4345034Z9?6
/?,
"?
input_1??????????
p 

 
? "????????????
,__inference_sequential_layer_call_fn_4349476Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
,__inference_sequential_layer_call_fn_4349485Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
%__inference_signature_wrapper_4346067?@?=
? 
6?3
1
input_1&?#
input_1??????????A"L?I
%
MDN?
MDN??????????
 
d?
d?????????