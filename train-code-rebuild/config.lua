
require("nn")

config = {}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

-- Training data
config.train_data = {}
config.train_data.file = paths.cwd() .. "/data/train.t7b"
config.train_data.alphabet = alphabet
config.train_data.length = 100
config.train_data.batch_size = 128

-- Validation data
config.val_data = {}
config.val_data.file =  paths.cwd() .. "/data/dev.t7b"
config.val_data.alphabet = alphabet
config.val_data.length = 100
config.val_data.batch_size = 128

-- The model
config.model = {}
-- #alphabet x 1014
config.model[1] = {module = "nn.LookupTable", char_vocab_size= #alphabet+1, inputFrameSize = 256}
config.model[2] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 4}
config.model[3] = {module = "nn.Threshold"}
config.model[4] = {module = "nn.TemporalMaxPooling", kW = 2, dW = 2}
-- 336 x 256
config.model[5] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 4}
config.model[6] = {module = "nn.Threshold"}
config.model[7] = {module = "nn.TemporalMaxPooling", kW = 2, dW = 2}
-- 110 x 256
config.model[8] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 2}
config.model[9] = {module = "nn.Threshold"}
-- 108 x 256
config.model[10] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 2}
config.model[11] = {module = "nn.Threshold"}
-- 106 x 256
config.model[12] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 2}
config.model[13] = {module = "nn.Threshold"}
-- 104 x 256
config.model[14] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 2}
config.model[15] = {module = "nn.Threshold"}
config.model[16] = {module = "nn.TemporalMaxPooling", kW = 2, dW = 2}
-- 34 x 256
config.model[17] = {module = "nn.Reshape", size = 2304}
-- 8704
config.model[18] = {module = "nn.Linear", inputSize = 2304, outputSize = 256}
config.model[19] = {module = "nn.Threshold"}
config.model[20] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[21] = {module = "nn.Linear", inputSize = 256, outputSize = 256}
config.model[22] = {module = "nn.Threshold"}
config.model[23] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[24] = {module = "nn.Linear", inputSize = 256, outputSize = 100}


config.model[25] = {module = "nn.Linear", inputSize = 200, outputSize = 3}
--config.model[25] = {module = "nn.Sigmoid"}
--config.model[26] = {module = "nn.Linear", inputSize = 100, outputSize = 3}
config.model[26] = {module = "nn.LogSoftMax"}


config.loss = nn.ClassNLLCriterion

-- The trainer
config.train = {}
local baseRate = 1e-2 * math.sqrt(config.train_data.batch_size) / math.sqrt(128)
config.train.rates = {[1] = baseRate/1,[15001] = baseRate/2,[30001] = baseRate/4,[45001] = baseRate/8,[60001] = baseRate/16,[75001] = baseRate/32,[90001]= baseRate/64,[105001] = baseRate/128,[120001] = baseRate/256,[135001] = baseRate/512,[150001] = baseRate/1024}
config.train.momentum = 0.9
config.train.decay = 1e-5


config.main = {}
config.main.dropout = true
config.main.type = "torch.DoubleTensor"
config.main.randomize = 5e-2
config.main.eras = 2
config.main.epoches = 100
config.main.test = true