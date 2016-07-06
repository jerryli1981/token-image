require("data")
require("model")
require("train")
require("test")

dofile("config.lua")

-- Prepare random number generator
math.randomseed(os.time())
torch.manualSeed(os.time())


main = {}

function main.main()
	main.argparse()
	main.new()
	main.run()
end

-- Parse arguments
function main.argparse()
   local cmd = torch.CmdLine()
   -- Options
   cmd:option("-debug",0, "debug setting")
   cmd:option("-gpu", 0, "using gpu")
   cmd:text()
   
   -- Parse the option
   local opt = cmd:parse(arg or {})   
   if opt.debug > 0 then
		dbg = require('debugger')
   end

	if opt.gpu > 0 then
   		require("cutorch")
   		require("cunn")
   		cutorch.setDevice(opt.gpu)
   		print("Device gpu set to ".. opt.gpu)
   	else
   		print("Device is cpu")
	end

end

function main.new()
	print("Loading datasets...")
	main.train_data = Data(config.train_data)
	main.val_data = Data(config.val_data)

	print("Loading the model...")
	main.model = Model(config.model, config.main.type)
	if config.main.randomize then
		main.model:randomize(config.main.randomize)
		print("Model randomized.")
	end
	main.model:type(config.main.type)
	print("Current model type: "..main.model:type())
	collectgarbage()

	print("Loading the trainer...")
	main.train = Train(main.train_data, main.model, config.loss(), config.train)

	print("Loading the tester...")
	main.test_val = Test(main.val_data, main.model, config.loss(), config.test)
end

function main.run()

	for i = 1, config.main.eras do
		if config.main.dropout then
			print("Enabling dropouts")
			main.model:enableDropouts()
		else
			print("Disabling dropouts")
			main.model:disableDropouts()
		end

		print("Training for era " .. i)
		main.train:run(config.main.epoches)

		if config.main.test == true then
			print("Disabling dropouts")
			main.model:disableDropouts()
			print("Testing on test data for era " .. i)
  			err = main.test_val:run()
  			print('Testing error is: ' .. err)
  		end
  	end
end

--Execute the main program
main.main()