
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <stdint.h>
#include <cassert>
#include "string.h"
#include <math.h>
#include <stdio.h>
#include "match.h"
#include "detections.h"
#include "mota_metrics.h"
#include "match_priv.h"
#include "log.h"

#define debugf if (1) log_warn

// Internal IOU pair helper
struct IOUPair {
    int det_id;
    int gt_id;
    double iou;
};

class MOTMetrics {
public:
    explicit MOTMetrics(double iou_threshold = 0.5)
        : iou_threshold_(iou_threshold) {}

    // C-compatible interface
    void addFrame(detection_t **dets, int num_dets,
                  detection_t **gts, int num_gts,
                  Cand *cands, int num_cands) {
        ++internal_.num_frames;

        // Track GT unique and appearance counts
        for (int i = 0; i < num_gts; ++i) {
            int gid = gts[i]->track_id;
            if (unique_objects_.insert(gid).second) {
                frame_count_[gid] = 0;
                match_count_[gid] = 0;
                last_matched_frame_[gid] = -1;
                last_det_match_[gid] = -1;
            }
            ++frame_count_[gid];
        }

        // Build det_ids and record all detection IDs
        std::vector<int> det_ids(num_dets);
        for (int i = 0; i < num_dets; ++i) {
            int did = dets[i]->track_id;
            det_ids[i] = did;
            all_det_ids_.insert(did);        // <-- track every observed det ID
        }

        std::vector<int> gt_ids(num_gts);
        for (int i = 0; i < num_gts; ++i) {
            gt_ids[i] = gts[i]->track_id;
        }

        // Convert Cand list to IOUPairs (thresholded)
        std::vector<IOUPair> ious;
        ious.reserve(num_cands);
        for (int i = 0; i < num_cands; ++i) {
            if (cands[i].score >= iou_threshold_) {
                IOUPair p;
                p.det_id = dets[cands[i].a]->track_id;
                p.gt_id  = gts[cands[i].b]->track_id;
                p.iou    = cands[i].score;
                ious.push_back(p);
            }
        }

        matchFrame(det_ids, gt_ids, ious);
    }

    // Compute and return results
    void compute(metric_results_t *out) {
        // Copy raw counts
        out->num_frames           = internal_.num_frames;
        out->num_matches          = internal_.num_matches;
        out->total_iou            = internal_.total_iou;
        out->num_false_positives  = internal_.num_false_positives;
        out->num_misses           = internal_.num_misses;
        out->num_switches         = internal_.num_switches;
        out->num_fragmentations   = internal_.num_fragmentations;

        out->num_unique_objects = static_cast<int>(unique_objects_.size());
        out->num_objects        = out->num_unique_objects;

        // Mostly tracked / partially / lost
        out->mostly_tracked    = 0;
        out->partially_tracked = 0;
        out->mostly_lost       = 0;
        for (int gid : unique_objects_) {
            int fcount = frame_count_[gid];
            int mcount = match_count_[gid];
            double ratio = (fcount > 0 ? static_cast<double>(mcount) / fcount : 0.0);
            if (ratio >= 0.8)      ++out->mostly_tracked;
            else if (ratio >= 0.2) ++out->partially_tracked;
            else                   ++out->mostly_lost;
        }

        // Derived detection metrics
        int total_gt = 0;
        for (auto &kv : frame_count_) total_gt += kv.second;
        out->recall    = (total_gt > 0 ? static_cast<double>(out->num_matches) / total_gt : 0.0);
        int total_det  = out->num_matches + out->num_false_positives;
        out->precision = (total_det > 0 ? static_cast<double>(out->num_matches) / total_det : 0.0);

        // MOTA & MOTP
        out->mota = 1.0 -
            (static_cast<double>(out->num_misses + out->num_false_positives + out->num_switches)
             / total_gt);
        out->motp = (out->num_matches > 0 ? out->total_iou / out->num_matches : 0.0);

        // Identity metrics
        out->idtp = out->num_matches;
        out->idfn = out->num_misses;
        out->idfp = out->num_false_positives;
        out->idr  = ((out->idtp + out->idfn) > 0
                     ? static_cast<double>(out->idtp) / (out->idtp + out->idfn)
                     : 0.0);
        out->idp  = ((out->idtp + out->idfp) > 0
                     ? static_cast<double>(out->idtp) / (out->idtp + out->idfp)
                     : 0.0);
        out->idf1 = ((out->idr + out->idp) > 0
                     ? 2 * out->idr * out->idp / (out->idr + out->idp)
                     : 0.0);

        // New metric: unique GT IDs that never matched
        int missed = 0;
        for (int gid : unique_objects_) {
            if (matched_gt_ids_.count(gid) == 0) ++missed;
        }
        out->missed = missed;

        // New metric: unique detection IDs that never matched
        int fp_tracks = 0;
        for (int did : all_det_ids_) {
            if (matched_det_ids_.count(did) == 0) ++fp_tracks;
        }
        out->fp_tracks = fp_tracks;
    }

private:
    struct Internal {
        int num_frames = 0;
        int num_matches = 0;
        int num_false_positives = 0;
        int num_misses = 0;
        int num_switches = 0;
        int num_fragmentations = 0;
        double total_iou = 0.0;
    } internal_;

    double iou_threshold_;

    // Original tracking
    std::unordered_set<int> unique_objects_;
    std::unordered_map<int,int> frame_count_;
    std::unordered_map<int,int> match_count_;
    std::unordered_map<int,int> last_matched_frame_;
    std::unordered_map<int,int> last_det_match_;

    // New containers
    std::unordered_set<int> matched_gt_ids_;  // GT IDs ever matched
    std::unordered_set<int> matched_det_ids_; // Det IDs ever matched
    std::unordered_set<int> all_det_ids_;     // All detected IDs seen

    void matchFrame(const std::vector<int>& det_ids,
                    const std::vector<int>& gt_ids,
                    const std::vector<IOUPair>& valid) {
        // Sort by descending IOU
        std::vector<IOUPair> sorted = valid;
        std::sort(sorted.begin(), sorted.end(),
                  [](auto &a, auto &b){ return a.iou > b.iou; });

        std::unordered_set<int> used_dets, used_gts;
        int frame_idx = internal_.num_frames;

        // Greedy matching loop
        for (auto &p : sorted) {
            if (!used_dets.count(p.det_id) && !used_gts.count(p.gt_id)) {
                used_dets.insert(p.det_id);
                used_gts.insert(p.gt_id);

                // Record that these tracks have been matched
                matched_det_ids_.insert(p.det_id);
                matched_gt_ids_.insert(p.gt_id);

                ++internal_.num_matches;
                internal_.total_iou += p.iou;
                ++match_count_[p.gt_id];

                int last_det = last_det_match_[p.gt_id];
                if (last_det >= 0 && last_det != p.det_id) {
                    ++internal_.num_switches;
                    if (frame_idx - last_matched_frame_[p.gt_id] > 1)
                        ++internal_.num_fragmentations;
                }
                last_det_match_[p.gt_id]     = p.det_id;
                last_matched_frame_[p.gt_id] = frame_idx;
            }
        }

        // Frame-level FPs & misses
        for (int id : det_ids)
            if (!used_dets.count(id)) ++internal_.num_false_positives;
        for (int id : gt_ids)
            if (!used_gts.count(id)) ++internal_.num_misses;
    }
};

struct mota_metrics
{
    MOTMetrics *m;
};

mota_metrics_t *mota_metrics_create()
{
    mota_metrics_t *mm=(mota_metrics_t *)malloc(sizeof(mota_metrics_t));
    memset(mm, 0, sizeof(mota_metrics_t));
    mm->m=new MOTMetrics;
    return mm;
}

static void generate_overlap_mask(detection_t *det)
{
    det->overlap_mask=box_to_8x8_mask(det->x0, det->y0, det->x1, det->y1);
}

static float box_iou(const detection_t *da, const detection_t *db)
{
    float x_left = fmaxf(da->x0, db->x0);
    float y_top = fmaxf(da->y0, db->y0);
    float x_right = fminf(da->x1, db->x1);
    float y_bottom = fminf(da->y1, db->y1);

    if (x_right < x_left || y_bottom < y_top)
        return 0.0f;

    float inter=(x_right - x_left) * (y_bottom - y_top);

    float area_a = (da->x1 - da->x0) * (da->y1 - da->y0);
    float area_b = (db->x1 - db->x0) * (db->y1 - db->y0);

    float union_area = area_a + area_b - inter;

    if (union_area <= 0.0f)
        return 0.0f;

    float ret=inter / union_area;
    return ret;
}

static float score_iou(const detection_t *da, const detection_t *db, float thr)
{
    float iou=box_iou(da, db);
    if (iou<thr) return 0.0;
    return iou;
}

void mota_metrics_add_frame(mota_metrics_t *mm, detection_t **dets, int num_dets, detection_t **gts, int num_gts)
{
    uint64_t mask_det[num_dets], mask_gt[num_gts];
    void *det_ptr[num_dets];
    void *gt_ptr[num_gts];

    for (int i = 0; i < num_dets; ++i) {
        generate_overlap_mask(dets[i]);
        mask_det[i] = dets[i]->overlap_mask;
        det_ptr[i]=(void*)dets[i];
    }
    for (int j = 0; j < num_gts; ++j) {
        generate_overlap_mask(gts[j]);
        mask_gt[j] = gts[j]->overlap_mask;
        gt_ptr[j]=(void*)gts[j];
    }

    // 1) Build sparse mask‐filtered list of candidate pairs
    Pair16 raw_cands[num_dets * num_gts];
    int num_raw = match_masks_optimized(
        mask_det, mask_gt,
        num_dets, num_gts,
        raw_cands
    );
    assert(num_raw <= num_dets * num_gts);

    // 2) Compute each pair's score
    Cand *all = (Cand *)malloc(sizeof(Cand) * num_raw);
    if (!all) return;
    for (int i = 0; i < num_raw; ++i) {
        uint16_t ia = raw_cands[i].a;
        uint16_t ib = raw_cands[i].b;
        float sc = score_iou((const detection_t *)det_ptr[ia], (const detection_t *)gt_ptr[ib], 0.45f);
        all[i].a     = ia;
        all[i].b     = ib;
        all[i].score = sc;
    }

    mm->m->addFrame(dets, num_dets, gts, num_gts, all, num_raw);
}

void mota_metrics_get_results(mota_metrics_t *mm, metric_results_t *out)
{
    assert(mm!=0);
    assert(out!=0);
    memset(out, 0, sizeof(metric_results_t));
    mm->m->compute(out);
}

void mota_metrics_destroy(mota_metrics_t *mm)
{
    delete mm->m;
    free(mm);
}
