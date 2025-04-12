#include "dataset.h"
#include <dirent.h> 
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "image.h"
#include "jpeg.h"
#include "detections.h"

struct dataset
{
    char image_path[1024];
    char label_path[1024];
    char ret_label[1024];
    char ret_image[1024];
    int num;
    char **names;
};

dataset_t *dataset_create(const char *path_to_images)
{
    dataset_t *ds=(dataset_t *)malloc(sizeof(dataset_t));
    assert(ds!=0);

    DIR *d;
    int num=0;
    struct dirent *dir;
    d = opendir(path_to_images);
    if (!d) return ds;
    while ((dir = readdir(d)) != NULL) 
    {
        if (dir->d_type == DT_REG) num++;
    }
    closedir(d);

    assert(strlen(path_to_images)<800);

    strcpy(ds->image_path, path_to_images);
    strcpy(ds->label_path, path_to_images);
    char *q=strstr(ds->label_path, "/images");
    assert(q!=0);
    *q=0;
    strcat(ds->label_path, "/labels");
    strcat(ds->label_path, strstr(ds->image_path, "/images")+7);

    printf("%s\n%s\n",ds->image_path,ds->label_path);

    ds->num=num;
    ds->names=(char **)malloc(sizeof(char*)*num);

    d = opendir(path_to_images);
    num=0;
    while (((dir = readdir(d)) != NULL)&&(num<ds->num))
    {
        if (dir->d_type == DT_REG)
        {
            if (strlen(dir->d_name)>4)
            {
                ds->names[num]=(char*)malloc(strlen(dir->d_name)+1);
                strcpy(ds->names[num], dir->d_name);
                char *p=strstr(ds->names[num], ".jpg");
                if (p==0)
                {
                    free(ds->names[num]);
                    continue;
                }
                *p=0;
                
                const char *label=dataset_get_label_path(ds, num);
                if (access(label, F_OK) != 0)
                {
                    free(ds->names[num]);
                    continue;
                }
                num++;
            }
        }
    }

    closedir(d);
    return ds;
}

void dataset_destroy(dataset_t *ds)
{
    if (!ds) return;
    for(int i=0;i<ds->num;i++) free(ds->names[i]);
    free(ds->names);
    free(ds);
}

int dataset_get_num(dataset_t *ds)
{
    if (!ds) return 0;
    return ds->num;
}

const char *dataset_get_image_path(dataset_t *ds, int index)
{
    if (!ds) return 0;
    if ((index<0) || index>=ds->num) return 0;
    strcpy(ds->ret_image, ds->image_path);
    strcat(ds->ret_image, "/");
    strcat(ds->ret_image, ds->names[index]);
    strcat(ds->ret_image, ".jpg");
    return ds->ret_image;
}

const char *dataset_get_label_path(dataset_t *ds, int index)
{
    if (!ds) return 0;
    if ((index<0) || index>=ds->num) return 0;
    strcpy(ds->ret_label, ds->label_path);
    strcat(ds->ret_label, "/");
    strcat(ds->ret_label, ds->names[index]);
    strcat(ds->ret_label, ".txt");
    return ds->ret_label;
}

image_t *dataset_get_image(dataset_t *ds, int index)
{
    const char *p=dataset_get_image_path(ds, index);
    return load_jpeg(p);
}

detections_t *dataset_get_gts(dataset_t *ds, int index)
{
    const char *p=dataset_get_label_path(ds, index);
    return load_detections(p);
}