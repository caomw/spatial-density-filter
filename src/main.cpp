/* 
 * Copyright (C) 2016 iCub Facility - Istituto Italiano di Tecnologia
 * Author: Ugo Pattacini
 * email:  ugo.pattacini@iit.it
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * http://www.robotcub.org/icub/license/gpl.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
*/

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <opencv2/opencv.hpp>

#include <yarp/os/ResourceFinder.h>
#include <yarp/os/Time.h>


/***********************************************/
class SpatialDensityFilter
{
public:
    /***********************************************/
    static std::vector<int> filter(const cv::Mat &data,
                                   const double radius,
                                   const int maxResults)
    {
        cv::flann::KDTreeIndexParams indexParams;
        cv::flann::Index kdtree(data,indexParams);

        cv::Mat query(1,data.cols,CV_32F);
        cv::Mat indices,dists;

        std::vector<int> res(data.rows);
        for (size_t i=0; i<res.size(); i++)
        {                        
            for (int c=0; c<query.cols; c++)
                query.at<float>(0,c)=data.at<float>(i,c);
            res[i]=kdtree.radiusSearch(query,indices,dists,
                                       radius,maxResults,
                                       cv::flann::SearchParams(128));
        }

        return res;
    }
};


/***********************************************/
int main(int argc, char *argv[])
{
    yarp::os::ResourceFinder rf;
    rf.setQuiet();    
    rf.setDefault("input-file","input.off");
    rf.setDefault("output-file","output.off");
    rf.setDefault("radius",yarp::os::Value(0.0002));
    rf.setDefault("nn-threshold",yarp::os::Value(80));
    rf.configure(argc,argv);

    double radius=rf.find("radius").asDouble();
    int nnThreshold=rf.find("nn-threshold").asInt();

    std::string fileName=rf.findFile("input-file");
    if (fileName.empty())
    {
        std::cerr<<"Unable to locate "
                 <<rf.find("input-file").asString()
                 <<std::endl;
        return 1;
    }

    std::ifstream fin;
    fin.open(fileName.c_str());
    if (!fin.is_open())
    {
        std::cerr<<"Unable to open "
                 <<rf.find("input-file").asString()
                 <<std::endl;
        return 2;
    }

    std::string offTag;
    int numVertices,dontcare;
    fin>>offTag>>numVertices;
    fin>>dontcare>>dontcare;

    cv::Mat data(numVertices,3,CV_32F);
    std::cout<<"#1 Reading input"<<std::endl;
    for (int i=0; i<numVertices; i++)
    {
        fin>>data.at<float>(i,0)
           >>data.at<float>(i,1)
           >>data.at<float>(i,2);
    }

    fin.close();

    std::cout<<"#2 Processing input"<<std::endl;
    double t0=yarp::os::Time::now();
    std::vector<int> res=SpatialDensityFilter::filter(data,radius,nnThreshold+1);
    double t1=yarp::os::Time::now();
    std::cout<<"Processed in "<<1e3*(t1-t0)<<" [ms]"<<std::endl;

    std::ofstream fout;
    fout.open(rf.find("output-file").asString().c_str());
    if (!fout.is_open())
    {
        std::cerr<<"Unable to open "
                 <<rf.find("output-file").asString()
                 <<std::endl;
        return 3;
    }

    std::cout<<"#3 Writing output"<<std::endl;
    fout<<"COFF"<<std::endl;
    fout<<numVertices<<" 0 0"<<std::endl;
    
    for (int i=0; i<numVertices; i++)
    {
        std::string color=res[i]>=nnThreshold?"0 255 0 0":"255 0 0 0";
        fout<<data.at<float>(i,0)<<" "
            <<data.at<float>(i,1)<<" "
            <<data.at<float>(i,2)<<" "
            <<color
            <<std::endl;
    }

    fout.close();

    return 0;
}

