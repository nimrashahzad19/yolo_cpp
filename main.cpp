#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;

std::string currentTimestamp() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_struct = std::localtime(&now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", tm_struct);
    return buffer;
}

void applyCartoonFilter(cv::Mat& frame) {
    cv::Mat gray, edges, color;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 7);
    cv::Laplacian(gray, edges, CV_8U, 5);
    cv::threshold(edges, edges, 80, 255, cv::THRESH_BINARY_INV);
    cv::bilateralFilter(frame, color, 9, 75, 75);
    cv::bitwise_and(color, color, frame, edges);
}

int main(int argc, char** argv) {
    std::ofstream log("log.txt");
    log << "Program started\n";

    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        MessageBoxA(NULL, "Failed to load Haar cascade", "Error", MB_OK);
        return -1;
    }

    cv::VideoCapture cap;
    cv::Mat inputImage;
    bool isImage = false;

    if (argc == 2) {
        std::string arg = argv[1];
        if (arg.find(".jpg") != std::string::npos ||
            arg.find(".png") != std::string::npos ||
            arg.find(".jpeg") != std::string::npos) {
            inputImage = cv::imread(arg);
            if (inputImage.empty()) {
                MessageBoxA(NULL, "Could not load image!", "Error", MB_OK);
                return -1;
            }
            isImage = true;
        }
        else {
            cap.open(arg);
        }
    }
    else {
        cap.open(0);
    }

    if (!isImage && !cap.isOpened()) {
        MessageBoxA(NULL, "Cannot open video or webcam", "Error", MB_OK);
        return -1;
    }

    fs::create_directory("snapshots");

    bool useGray = false;
    bool useBlur = false;
    bool useEdge = false;
    bool useCartoon = false;

    MessageBoxA(NULL,
        "Controls:\n"
        "[G] Toggle Grayscale\n"
        "[B] Toggle Blur\n"
        "[E] Toggle Edge\n"
        "[C] Toggle Cartoon\n"
        "[S] Save Snapshot\n"
        "[ESC] Exit",
        "Instructions", MB_OK);

    while (true) {
        cv::Mat frame;

        if (isImage) {
            frame = inputImage.clone();
        }
        else {
            cap >> frame;
            if (frame.empty()) {
                log << "Empty frame detected. Skipping...\n";
                continue;
            }
        }

        cv::Mat processed = frame.clone();

        if (useGray) cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
        if (useBlur) cv::GaussianBlur(processed, processed, cv::Size(15, 15), 0);
        if (useEdge) {
            cv::Mat gray, edges;
            cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
            cv::Canny(gray, edges, 100, 200);
            cv::cvtColor(edges, processed, cv::COLOR_GRAY2BGR);
        }
        if (useCartoon) applyCartoonFilter(processed);

        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(processed, faces, 1.1, 4);
        for (const auto& face : faces) {
            cv::rectangle(processed, face, cv::Scalar(0, 255, 0), 2);
        }

        std::string label = "Faces: " + std::to_string(faces.size());
        cv::putText(processed, label, { 10, 30 }, cv::FONT_HERSHEY_SIMPLEX, 1.0, { 255, 255, 255 }, 2);

        // Resize for better display experience
        int displayHeight = 720;
        int displayWidth = static_cast<int>((double)processed.cols * displayHeight / processed.rows);
        cv::resize(processed, processed, cv::Size(displayWidth, displayHeight));

        cv::namedWindow("Face Detection", cv::WINDOW_NORMAL);
        cv::imshow("Face Detection", processed);

        char key = static_cast<char>(cv::waitKey(isImage ? 0 : 30));

        if (key == 27) {
            log << "ESC pressed. Exiting.\n";
            break;
        }
        else if (key == 'g') {
            useGray = !useGray;
            log << "Grayscale: " << (useGray ? "ON" : "OFF") << "\n";
        }
        else if (key == 'b') {
            useBlur = !useBlur;
            log << "Blur: " << (useBlur ? "ON" : "OFF") << "\n";
        }
        else if (key == 'e') {
            useEdge = !useEdge;
            log << "Edge: " << (useEdge ? "ON" : "OFF") << "\n";
        }
        else if (key == 'c') {
            useCartoon = !useCartoon;
            log << "Cartoon: " << (useCartoon ? "ON" : "OFF") << "\n";
        }
        else if (key == 's') {
            std::string baseName;

            if (argc == 2) {
                std::string inputName = argv[1];
                size_t lastDot = inputName.find_last_of('.');
                size_t lastSlash = inputName.find_last_of("/\\");
                std::string nameOnly = inputName.substr(lastSlash + 1, lastDot - lastSlash - 1);
                baseName = nameOnly + "_result_" + currentTimestamp();
            }
            else {
                baseName = "webcam_result_" + currentTimestamp();
            }

            std::string filename = "snapshots/" + baseName + ".png";
            cv::imwrite(filename, processed);
            log << "Snapshot saved: " << filename << "\n";
            MessageBoxA(NULL, ("Snapshot saved to " + filename).c_str(), "Saved", MB_OK);
        }


        if (isImage && key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    log << "Program ended\n";
    return 0;
}
