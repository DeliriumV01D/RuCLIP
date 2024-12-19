#include "TorchHeader.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <string>
#include <regex>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "RuCLIP.h"
#include "RuCLIPProcessor.h"

///
bool MatchImageAndText(torch::Device& device, CLIP& clip, RuCLIPProcessor& processor,
	const std::vector<cv::Mat>& images, const std::vector<std::string>& labels)
{
	bool res = false;

	auto dummy_input = processor(labels, images);
	try {
		torch::Tensor logits_per_image = clip->forward(dummy_input.first.to(device), dummy_input.second.to(device));
		torch::Tensor logits_per_text = logits_per_image.t();
		auto probs = logits_per_image.softmax(/*dim = */-1).detach().cpu();
		std::cout << "probs per image: " << probs << std::endl;
		res = true;
	}
	catch (std::exception& e)
	{
		res = false;
		std::cerr << e.what() << std::endl;
	}
	return res;
}

///
bool MatchOne2ManyImages(torch::Device& device, CLIP& clip, RuCLIPProcessor& processor,
	const std::vector<cv::Mat>& images, const std::vector<std::string>& labels)
{
	bool res = false;

	if (images.size() < 2)
	{
		std::cerr << "There must be at least 2 images: one for reference and others to search through" << std::endl;
		return res;
	}

	try
	{
		std::cout << "Create tensor for reference..." << std::endl;
		torch::Tensor embed = processor.EncodeImage(images[0]);
		embed = embed / embed.norm(2/*L2*/, -1, true);

		std::cout << "Create tensor for others..." << std::endl;
		std::vector<torch::Tensor> imgsTensors;
		imgsTensors.reserve(images.size() - 1);
		for (size_t i = 1; i < images.size(); ++i)
		{
			imgsTensors.emplace_back(processor.EncodeImage(images[i]));
		}
		auto imgsFeatures = torch::stack(imgsTensors).to(device);
		imgsFeatures = imgsFeatures / imgsFeatures.norm(2/*L2*/, -1, true);

		std::cout << "Create tensor for text..." << std::endl;
		std::vector<torch::Tensor> textTensors;
		textTensors.reserve(labels.size());
		for (auto label : labels)
		{
			textTensors.emplace_back(processor.EncodeText(label));
		}
		auto textFeatures = clip->EncodeText(torch::stack(textTensors).to(device));
		//auto textFeatures = torch::stack(textTensors).to(device);
		textFeatures = textFeatures / textFeatures.norm(2/*L2*/, -1, true);

		std::cout << "Relevancy: one image to " << (images.size() - 1) << " with " << labels.size() << " negatives" << std::endl;
		torch::Tensor probs = Relevancy(embed, imgsFeatures, textFeatures);
		std::cout << "Probs for image2images: " << probs << std::endl;
		res = true;
	}
	catch (std::exception& e)
	{
		res = false;
		std::cerr << e.what() << std::endl;
	}

	return res;
}


///
int main(int argc, const char* argv[])
{
	setlocale(LC_ALL, "");

	const char* keys =
	{
		"{ test_ind         |                    | 0: matching images and text, 1: matching first image with all others (text - negative embeddings) | }"
		"{ imgs             |img1.jpg,img2.jpg   | List of images | }"
		"{ text             |cat,bear,fox        | List of labels | }"
		"{ clip             |../data/ruclip-vit-large-patch14-336 | Path to RuClip model | }"
		"{ bpe              |../data/ruclip-vit-large-patch14-336/bpe.model | Path to tokenizer | }"
		"{ img_size         |336                 | Input model size | }"
	};
	
	cv::CommandLineParser parser(argc, argv, keys);
	parser.printMessage();

	int testInd = parser.get<int>("test_ind"); 
	std::string imagesStr = parser.get<std::string>("imgs");
	std::string labelsStr = parser.get<std::string>("text");
	std::string pathToClip = parser.get<std::string>("clip");
	std::string pathToBPE = parser.get<std::string>("bpe");
	int INPUT_IMG_SIZE = parser.get<int>("img_size");

	torch::manual_seed(24);

	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Running on GPU." << std::endl;
		device = torch::Device(torch::kCUDA);
	}	else {
		std::cout << "CUDA is not available! Running on CPU." << std::endl;
	}

	std::cout << "Load clip from: " << pathToClip << std::endl;
	CLIP clip = FromPretrained(pathToClip);
	clip->to(device);

	std::cout << "Load processor from: " << pathToBPE << std::endl;
	RuCLIPProcessor processor(
		pathToBPE,
		INPUT_IMG_SIZE,
		77,
		{ 0.48145466, 0.4578275, 0.40821073 },
		{ 0.26862954, 0.26130258, 0.27577711 }
	);

	std::vector<cv::Mat> images;
	{
		std::cout << "images: " << std::endl;
		std::regex sep("[,]+");
		std::sregex_token_iterator tokens(imagesStr.cbegin(), imagesStr.cend(), sep, -1);
		std::sregex_token_iterator end;
		for (; tokens != end; ++tokens)
		{
			cv::Mat img = cv::imread(*tokens, cv::IMREAD_COLOR);

			std::cout << (*tokens) << " is loaded: " << !img.empty() << std::endl;

			cv::resize(img, img, cv::Size(INPUT_IMG_SIZE, INPUT_IMG_SIZE), cv::INTER_CUBIC);

			images.emplace_back(img);
		}
	}

	//Завести метки
	std::vector<std::string> labels;
	{
		std::cout << "labels: ";
		std::regex sep("[,]+");
		std::sregex_token_iterator tokens(labelsStr.cbegin(), labelsStr.cend(), sep, -1);
		std::sregex_token_iterator end;
		for (; tokens != end; ++tokens)
		{
			std::cout << (*tokens) << " | ";
			labels.push_back(*tokens);
		}
		std::cout << std::endl;
	}

	std::cout << "Running..." << std::endl;

	switch (testInd)
	{
	case 0:
		MatchImageAndText(device, clip, processor, images, labels);
		break;

	case 1:
		MatchOne2ManyImages(device, clip, processor, images, labels);
		break;

	default:
		std::cerr << "Wrong test index: " << testInd << std::endl;
	}
	
	std::cout << "The end!" << std::endl;
}