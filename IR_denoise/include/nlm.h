#ifndef _NLM_H_
#define _NLM_H_

#include "IR_denoise.h"

class NLM : public IR_denoise
{
    public:
        NLM(int t, int s, float h) : template_size(t), search_size(s), h(h)
        {
            if (template_size % 2 == 0)
            {
                std::cout << "Error: Template size must be odd!" << std::endl;
                exit(1);
            }
            if (search_size % 2 == 0)
            {
                std::cout << "Error: Search size must be odd!" << std::endl;
                exit(1);
            }
        }
        virtual ~NLM(){
            std::cout<<"NLM destructor"<<std::endl;
        };
        void denoise (cv::Mat& dst) const;
        void set_parametres(int t, int s, float h);

        // void fastNonLocalMeans(cv::Mat& src, cv::Mat& dst) const;

    private:
        int template_size;
        int search_size;
        float h;
        float distance(const cv::Mat& template_img, const cv::Mat& patch) const;
};


inline void NLM::set_parametres(int t, int s, float h)
{
    if (t % 2 == 0)
    {
        std::cout << "Error: Template size must be odd!" << std::endl;
        exit(1);
    }
    if (s % 2 == 0)
    {
        std::cout << "Error: Search size must be odd!" << std::endl;
        exit(1);
    }
    this->template_size = t;
    this->search_size = s;
    this->h = h;
}
#endif

