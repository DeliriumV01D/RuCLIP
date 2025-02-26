# RuCLIP
Unofficial c++ LibTorch implementation of RuCLIP (Sber AI)

RuCLIP (Russian Contrastive Language–Image Pretraining) is a multimodal model for obtaining images and text similarities and rearranging captions and pictures.

Original PyTorch code: https://github.com/ai-forever/ru-clip

Original CLIP OpenAI paper: https://arxiv.org/pdf/2103.00020.pdf

#### Dependencies: 
[libTorch](https://pytorch.org), 
[YouTokenToMe tokenizer](https://github.com/VKCOM/YouTokenToMe), 
[OpenCV](https://opencv.org/releases/), 
[nlohmann json](https://github.com/nlohmann/json) 

#### Test

Test images:

![1](https://github.com/DeliriumV01D/RuCLIP/assets/46240032/2a006f77-30c8-45a7-b2c5-928899f2db8a)
![2](https://github.com/DeliriumV01D/RuCLIP/assets/46240032/2cd7364c-6368-4658-a2cb-c3350e8abfa7)
![3](https://github.com/DeliriumV01D/RuCLIP/assets/46240032/aea25ee6-a9a6-4fd8-b435-dc8a4c17b009)

Test labels:
{"кот", "медведь", "лиса"}

RuCLIP probabilities:

	0.8879  0.0063  0.1058
	0.0014  0.0026  0.9960
	0.0002  0.9994  0.0003

For minimal example see main.cpp:

	CLIP clip = FromPretrained("../data/ruclip-vit-large-patch14-336");
	clip->to(device);

	RuCLIPProcessor processor(
		"../data/ruclip-vit-large-patch14-336/bpe.model",
		INPUT_IMG_SIZE,
		77,
		{ 0.48145466, 0.4578275, 0.40821073 },
		{ 0.26862954, 0.26130258, 0.27577711 }
	);

	//Загрузить картинки
	std::vector <cv::Mat> images;
	images.push_back(cv::imread("../data//test_images/1.png", cv::ImreadModes::IMREAD_COLOR));
	images.push_back(cv::imread("../data//test_images/2.jpg", cv::ImreadModes::IMREAD_COLOR));
	images.push_back(cv::imread("../data//test_images/3.jpg", cv::ImreadModes::IMREAD_COLOR));
	//resize->[336, 336]
	for (auto &it : images)
		 cv::resize(it, it, cv::Size(INPUT_IMG_SIZE, INPUT_IMG_SIZE));

	//Завести метки
	std::vector<std::string> labels;
	labels = {"кот", "медведь", "лиса"};

	auto dummy_input = processor(labels, images);
	try {
		torch::Tensor logits_per_image = clip->forward(dummy_input.first.to(device), dummy_input.second.to(device));
		torch::Tensor logits_per_text = logits_per_image.t();
		auto probs = logits_per_image.softmax(/*dim = */-1).detach().cpu();
		std::cout << "probs per image: " << probs << std::endl;

  To load weights you need to export checkpoint to jit format use [ai-forever/ru-clip](https://github.com/ai-forever/ru-clip):
<details>

	import ruclip
	import torch
	import base64
	import requests
	from PIL import Image
	from io import BytesIO
	import matplotlib.pyplot as plt

	def save_clip_model(clip_name:str, size:int, device:str):
		clip, processor = ruclip.load(clip_name, device=device)

		# prepare images
		bs4_urls = requests.get('https://raw.githubusercontent.com/ai-forever/ru-dolph/master/pics/pipelines/cats_vs_dogs_bs4.json').json()
		images = [Image.open(BytesIO(base64.b64decode(bs4_url))) for bs4_url in bs4_urls]

		# prepare classes
		classes = ['кошка', 'собака']
		templates = ['{}', 'это {}', 'на картинке {}', 'это {}, домашнее животное']

		# predict
		predictor = ruclip.Predictor(clip, processor, device, bs=8, templates=templates)
		with torch.no_grad():
			text_latents = predictor.get_text_latents(classes)
			pred_labels = predictor.run(images, text_latents)

		# show results
		f, ax = plt.subplots(2,4, figsize=(12,6))
		for i, (pil_img, pred_label) in enumerate(zip(images, pred_labels)):
			ax[i//4, i%4].imshow(pil_img)
			ax[i//4, i%4].set_title(classes[pred_label])
		plt.show()

    		# export to jit
		var=(torch.ones((1, 77)).long(), torch.ones((1, 3, size, size)))  
		traced_script_module = torch.jit.trace(clip, var)
		traced_script_module.save("saves/" + clip_name + ".zip")

	#device = 'cuda'
	device = 'cpu'

	clip_name = 'ruclip-vit-large-patch14-336'
	size = 336
	save_clip_model(clip_name, size, device)
</details>

#### Run a default example

On Windows with CMake build can be some problems:
1. nvToolsExt not found with CUDA 12.1. You can set include directory "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include/nvtx3" and any library
2. After CMake generate step remove from linker dependency nvToolsExt library - they are not used in project and nvToolsExt64_1.dll has alredy in libTorch


**Run example matching images to text with bat file:**

            set IMGS=C:\work\clip\ruclip_\CLIP\data\test_images\1.png,C:\work\clip\ruclip_\CLIP\data\test_images\2.jpg,C:\work\clip\ruclip_\CLIP\data\test_images\3.jpg
            set LABELS=cat,bear,fox

            set CLIP=C:\work\clip\ruclip_\CLIP\data\ruclip-vit-large-patch14-336

            RuCLIP.exe --test_ind=0 --imgs=%IMGS% --text=%LABELS% --clip=%CLIP%

**Run example matching one image to another with bat file:**

            set IMGS=C:\work\clip\ruclip_\CLIP\data\test_images\bag2_1.jpg,C:\work\clip\ruclip_\CLIP\data\test_images\bag0_0.jpg,C:\work\clip\ruclip_\CLIP\data\test_images\bag1_0.jpg,C:\work\clip\ruclip_\CLIP\data\test_images\bag2_0.jpg,C:\work\clip\ruclip_\CLIP\data\test_images\bag2_2.jpg,C:\work\clip\ruclip_\CLIP\data\test_images\bag5_2.jpg
            set LABELS=person,car,truck,bus

            set CLIP=C:\work\clip\ruclip_\CLIP\data\ruclip-vit-large-patch14-336

            RuCLIP.exe --test_ind=1 --imgs=%IMGS% --text=%LABELS% --clip=%CLIP%






