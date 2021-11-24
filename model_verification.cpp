#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
using namespace cv;
using namespace std;

vector<string> _csv(string s)
    {
    vector<string> arr;
    istringstream delim(s);
    string token;
    int c = 0;
    while (getline(delim, token, ','))        
    {
        arr.push_back(token);                
        c++;                                           
    }
    return  arr;
}

cv::Mat _datacsv(string s)
    {
    cv::Mat arr;
    istringstream delim(s);
    string token;
    int c = 0;
    while (getline(delim, token, ','))        
    {   
        arr.push_back((float)std::stof(token));           
        c++;                                           
    }
    return arr;
}

int main()
{   
    ifstream inFile("/home/tobyc/c_code/weights.csv", ios::in);
    ifstream dataFile("/home/tobyc/c_code/test_array.csv", ios::in);
    fstream f;
    if (!inFile)
    {
        cout << "No such file" << endl;
        exit(1);
    }
    string data_line;
    cv::Mat data;
    //cv::Mat input_data = (cv::Mat_<float>(18, 1)<<0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00279888, 0., 0., 0., 0.08843967, 0., 0., 0.08796605); 
    cv::Mat dense_0 = cv::Mat_<float>(18, 256);
    cv::Mat batch_normalization_0 = cv::Mat_<float>(4, 256);
    cv::Mat dense_1 = cv::Mat_<float>(256, 128);
    cv::Mat batch_normalization_1 = cv::Mat_<float>(4, 128);
    cv::Mat dense_2 = cv::Mat_<float>(128, 64);
    cv::Mat batch_normalization_2 = cv::Mat_<float>(4, 64);
    cv::Mat dense_3 = cv::Mat_<float>(64, 32);
    cv::Mat batch_normalization_3 = cv::Mat_<float>(4, 32);
    cv::Mat dense_4 = cv::Mat_<float>(32, 1);

    int dense_0_num = 0;
    int batch_normalization_0_num = 0;
    int dense_1_num = 0;
    int batch_normalization_1_num = 0;
    int dense_2_num = 0;
    int batch_normalization_2_num = 0;
    int dense_3_num = 0;
    int batch_normalization_3_num = 0;
    int dense_4_num = 0;
    string line;
    while (getline(inFile, line))
    {  
        vector<string> a = _csv(line);
        if(a[0] == "dense")
        {
            for (int ii = 1; ii < a.size(); ii++)
                    {
                        dense_0.at<float>(dense_0_num, ii-1) = (float)std::stof(a[ii]);
                    }
            dense_0_num++;
        }
        else if(a[0] == "batch_normalization"){
            for (int ii = 1; ii < a.size(); ii++)
                    {
                        batch_normalization_0.at<float>(batch_normalization_0_num, ii-1) = (float)std::stof(a[ii]);
                    }
            batch_normalization_0_num++;
        }
        else if(a[0] == "dense_1")
        {
            for (int ii = 1; ii < a.size(); ii++)
                    {   
                        dense_1.at<float>(dense_1_num, ii-1) = (float)std::stof(a[ii]);
                    }
            dense_1_num++;
        }
        else if(a[0] == "batch_normalization_1"){
            for (int ii = 1; ii < a.size(); ii++)
                    {   
                        batch_normalization_1.at<float>(batch_normalization_1_num, ii-1) = (float)std::stof(a[ii]);
                    }
            batch_normalization_1_num++;
        }
        else if(a[0] == "dense_2")
        {
            for (int ii = 1; ii < a.size(); ii++)
                    {   
                        dense_2.at<float>(dense_2_num, ii-1) = (float)std::stof(a[ii]);
                    }
            dense_2_num++;
        }
        else if(a[0] == "batch_normalization_2"){
            for (int ii = 1; ii < a.size(); ii++)
                    {   
                        batch_normalization_2.at<float>(batch_normalization_2_num, ii-1) = (float)std::stof(a[ii]);
                    }
            batch_normalization_2_num++;
        }
        else if(a[0] == "dense_3")
        {
            for (int ii = 1; ii < a.size(); ii++)
                    {   
                        dense_3.at<float>(dense_3_num, ii-1) = (float)std::stof(a[ii]);
                    }
            dense_3_num++;
        }
        else if(a[0] == "batch_normalization_3"){
            for (int ii = 1; ii < a.size(); ii++)
                    {
                        batch_normalization_3.at<float>(batch_normalization_3_num, ii-1) = (float)std::stof(a[ii]);
                    }
            batch_normalization_3_num++;
        }
        else if(a[0] == "dense_4")
        {
            for (int ii = 1; ii < a.size(); ii++)
                    {   
                        dense_4.at<float>(dense_4_num, ii-1) = (float)std::stof(a[ii]);
                    }
            dense_4_num++;
        }
    }
    f.open("result_c.csv", ios::out);
    while (getline(dataFile, data_line)){
        //cout << data_line << endl;
        cv::Mat d = _datacsv(data_line);
        //cout << d.t() << endl;
        cv::Mat first_output(256, 1, CV_64F);
        cv::Mat second_output(128, 1, CV_64F);
        cv::Mat third_output(64, 1, CV_64F);
        cv::Mat fourth_output(32, 1, CV_64F);
        cv::Mat fifth_output(1, 1, CV_64F);

        first_output = d.t() * dense_0;
        first_output.setTo(0, first_output<=0);
        cv::Mat denominator = batch_normalization_0.row(3) + cv::Scalar(0.001);
        cv::sqrt(denominator, denominator);
        first_output = (batch_normalization_0.row(0).mul(first_output - batch_normalization_0.row(2))/denominator) + batch_normalization_0.row(1);
        
        denominator = batch_normalization_1.row(3) + cv::Scalar(0.001);
        cv::sqrt(denominator, denominator);
        second_output = first_output * dense_1;
        second_output.setTo(0, second_output<=0);
        second_output = (batch_normalization_1.row(0).mul(second_output - batch_normalization_1.row(2))/denominator) + batch_normalization_1.row(1);
        
        denominator = batch_normalization_2.row(3) + cv::Scalar(0.001);
        cv::sqrt(denominator, denominator);
        third_output = second_output * dense_2;
        third_output.setTo(0, third_output<=0);
        third_output = (batch_normalization_2.row(0).mul(third_output - batch_normalization_2.row(2))/denominator) + batch_normalization_2.row(1);

        denominator = batch_normalization_3.row(3) + cv::Scalar(0.001);
        cv::sqrt(denominator, denominator);
        fourth_output = third_output * dense_3;
        fourth_output.setTo(0, fourth_output<=0);
        fourth_output = (batch_normalization_3.row(0).mul(fourth_output - batch_normalization_3.row(2))/denominator) + batch_normalization_3.row(1);

        fifth_output = fourth_output * dense_4;
        fifth_output.setTo(0, fifth_output<=0);
        cout << fifth_output.at<float>(0,0) << endl;
        f << fifth_output.at<float>(0,0) << endl;

        //cv::waitKey(0);
    }
    f.close();
    
    return 0;
}
