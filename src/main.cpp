#include "TorchHeader.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <regex>

#include <opencv2/opencv.hpp>

#include "RuCLIP.h"
#include "RuCLIPProcessor.h"

///
bool MatchImageAndText(torch::Device& device, CLIP& clip, RuCLIPProcessor& processor,
	const std::vector<cv::Mat>& images, const std::vector<std::string>& labels,
	bool showResult)
{
	bool res = false;

	auto dummy_input = processor(labels, images);
	try {
		torch::Tensor logits_per_image = clip->forward(dummy_input.first.to(device), dummy_input.second.to(device));
		torch::Tensor logits_per_text = logits_per_image.t();
		auto probs = logits_per_image.softmax(/*dim = */-1).detach().cpu();
		std::cout << "probs per image:\n" << probs << std::endl;
		
		if (showResult)
		{
			for (size_t row = 0; row < images.size(); ++row)
			{
				std::string wndName = std::to_string(row);

				float maxProb = 0.f;
				size_t maxCol = 0;
				for (size_t col = 0; col < labels.size(); ++col)
				{
					float value = probs[row][col].item<float>();
					if (maxProb < value)
					{
						maxProb = value;
						maxCol = col;
					}
				}
				std::string label = labels[maxCol] + ": " + std::to_string(maxProb);

				int baseLine = 0;
				double fontScale = (images[row].cols < 1920) ? 0.7 : 2.0;
				int thickness = (images[row].cols < 1920) ? 1 : 2;
				int lineType = cv::LINE_AA;
				int fontFace = cv::FONT_HERSHEY_SCRIPT_COMPLEX;
				cv::Size labelSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseLine);
				cv::putText(images[row], label, cv::Point(2, 2 + labelSize.height), fontFace, fontScale, cv::Scalar(255, 0, 255), thickness, lineType);


				cv::imshow(wndName, images[row]);
				cv::waitKey(1);
			}
			cv::waitKey(0);
		}
		
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
	const std::vector<cv::Mat>& images, const std::vector<std::string>& labels,
	bool showResult)
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
		torch::Tensor embed = clip->EncodeImage(processor.EncodeImage(images[0]).to(device));
		embed = embed / embed.norm(2/*L2*/, -1, true);
		std::cout << "embed: " << embed.sizes() << std::endl;

		std::cout << "Create tensor for others..." << std::endl;
		std::vector<torch::Tensor> imgsTensors;
		imgsTensors.reserve(images.size() - 1);
		for (size_t i = 1; i < images.size(); ++i)
		{
			imgsTensors.emplace_back(processor.EncodeImage(images[i]));
			std::cout << "imgsTensors: " << imgsTensors.back().sizes() << std::endl;
		}
		auto toStack = torch::stack(imgsTensors);
		std::cout << "toStack: " << toStack.sizes() << std::endl;
		auto toSqueeze = toStack.squeeze(1);
		std::cout << "toSqueeze: " << toSqueeze.sizes() << std::endl;
		auto imgsFeatures = clip->EncodeImage(toSqueeze.to(device));
		imgsFeatures = imgsFeatures / imgsFeatures.norm(2/*L2*/, -1, true);
		std::cout << "imgsFeatures: " << imgsFeatures.sizes() << std::endl;

		std::cout << "Create tensor for text..." << std::endl;
		std::vector<torch::Tensor> textTensors;
		textTensors.reserve(labels.size());
		for (auto label : labels)
		{
			textTensors.emplace_back(processor.EncodeText(label));
		}
		auto textFeatures = clip->EncodeText(torch::stack(textTensors).to(device));
		textFeatures = textFeatures / textFeatures.norm(2/*L2*/, -1, true);

		std::cout << "embed: " << embed.sizes() << ", imgsFeatures: " << imgsFeatures.sizes() << ", textFeatures: " << textFeatures.sizes() << std::endl;

		std::cout << "Relevancy: one image to " << (images.size() - 1) << " with " << labels.size() << " negatives" << std::endl;
		torch::Tensor probs = Relevancy(embed, imgsFeatures, textFeatures);
		std::cout << "Probs for image2images:\n" << probs << std::endl;
		
		
		if (showResult)
		{
			size_t row = 0;
			cv::imshow("source", images[row]);
			cv::waitKey(1);

			std::string wndName = std::to_string(row);

			for (size_t col = 1; col < images.size(); ++col)
			{
				float prob = probs[row][col - 1].item<float>();
				std::string label = std::to_string(col) + ": " + std::to_string(prob);

				int baseLine = 0;
				double fontScale = (images[col].cols < 1920) ? 0.7 : 2.0;
				int thickness = (images[col].cols < 1920) ? 1 : 2;
				int lineType = cv::LINE_AA;
				int fontFace = cv::FONT_HERSHEY_SCRIPT_COMPLEX;
				cv::Size labelSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseLine);
				cv::putText(images[col], label, cv::Point(2, 2 + labelSize.height), fontFace, fontScale, cv::Scalar(255, 0, 255), thickness, lineType);

				cv::imshow(wndName, images[col]);
				cv::waitKey(1);
			}

			cv::waitKey(0);
		}
		
		
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
		"{ test_ind         |0                   | 0: matching images and text, 1: matching first image with all others (text - negative embeddings) | }"
		"{ imgs             |img1.jpg,img2.jpg   | List of images | }"
		"{ text             |cat,bear,fox        | List of labels | }"
		"{ clip             |../data/ruclip-vit-large-patch14-336 | Path to RuClip model | }"
		"{ show             |1                   | Show result | }"
	};
	
	cv::CommandLineParser parser(argc, argv, keys);
	parser.printMessage();

	int testInd = parser.get<int>("test_ind"); 
	std::string imagesStr = parser.get<std::string>("imgs");
	std::string labelsStr = parser.get<std::string>("text");
	std::string pathToClip = parser.get<std::string>("clip");
	bool showResult = parser.get<int>("show") != 0;

	torch::manual_seed(24);

	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Running on GPU." << std::endl;
		device = torch::Device(torch::kCUDA);
	} else {
		std::cout << "CUDA is not available! Running on CPU." << std::endl;
	}

	std::cout << "Load clip from: " << pathToClip << std::endl;
	CLIP clip{ FromPretrained(pathToClip) };
	clip->to(device);

	RuCLIPProcessor processor{ RuCLIPProcessor::FromPretrained(pathToClip) };

	std::vector<cv::Mat> images;
	{
		std::cout << "images: " << std::endl;
		std::regex sep("[,]+");
		std::sregex_token_iterator tokens(imagesStr.cbegin(), imagesStr.cend(), sep, -1);
		std::sregex_token_iterator end;
		for (; tokens != end; ++tokens)
		{
			cv::Mat img = cv::imread(*tokens, cv::IMREAD_COLOR);

			std::cout << (*tokens) << " is loaded: " << !img.empty() << " with size " << img.size() << std::endl;
			std::cout << "Resizing to " << cv::Size(processor.GetImageSize(), processor.GetImageSize()) << "..." << std::endl;
			cv::resize(img, img, cv::Size(processor.GetImageSize(), processor.GetImageSize()), cv::INTER_CUBIC);

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
		MatchImageAndText(device, clip, processor, images, labels, showResult);
		break;

	case 1:
		MatchOne2ManyImages(device, clip, processor, images, labels, showResult);
		break;

	default:
		std::cerr << "Wrong test index: " << testInd << std::endl;
	}
	
	std::cout << "The end!" << std::endl;
}