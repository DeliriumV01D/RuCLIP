#include "TorchHeader.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

#include "RuCLIP.h"
#include "RuCLIPProcessor.h"

int main(int argc, const char* argv[])
{
	setlocale(LC_ALL, "");
	const int INPUT_IMG_SIZE = 336;
	torch::manual_seed(24);

	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Running on GPU." << std::endl;
		device = torch::Device(torch::kCUDA);
	}	else {
		std::cout << "CUDA is not available! Running on CPU." << std::endl;
	}

	CLIP clip = FromPretrained("..//data//ruclip-vit-large-patch14-336");
	clip->to(device);

	RuCLIPProcessor processor(
		"..//data//ruclip-vit-large-patch14-336//bpe.model",
		INPUT_IMG_SIZE,
		77,
		{ 0.48145466, 0.4578275, 0.40821073 },
		{ 0.26862954, 0.26130258, 0.27577711 }
	);

		////Или можно без него сначала попробовать
		//RuCLIPPredictor(clip, processor, device, templates, 8);

	//Загрузить картинки
	std::vector <cv::Mat> images;
	images.push_back(cv::imread("..//data//test_images//1.png", cv::ImreadModes::IMREAD_COLOR));
	images.push_back(cv::imread("..//data//test_images//2.jpg", cv::ImreadModes::IMREAD_COLOR));
	images.push_back(cv::imread("..//data//test_images//3.jpg", cv::ImreadModes::IMREAD_COLOR));
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
	}	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
}