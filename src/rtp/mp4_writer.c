// mp4_writer.hpp / .cpp (single file ok)
// C++17, zero external deps.
// Build: g++ -O3 -std=c++17 mp4_writer.cpp -o demo

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <limits>

typedef struct mp4_writer mp4_writer_t;

// -------------------------- Small utils -------------------------------------
static void mp4w_die(const char* m){ throw std::runtime_error(m); }
static void mp4w_die(const std::string& m){ throw std::runtime_error(m); }

struct Buf {
    std::vector<uint8_t> v;
    size_t tell() const { return v.size(); }
    void u8(uint8_t x){ v.push_back(x); }
    void be16(uint16_t x){ v.push_back((x>>8)&0xFF); v.push_back(x&0xFF); }
    void be24(uint32_t x){ v.push_back((x>>16)&0xFF); v.push_back((x>>8)&0xFF); v.push_back(x&0xFF); }
    void be32(uint32_t x){ v.push_back((x>>24)&0xFF); v.push_back((x>>16)&0xFF); v.push_back((x>>8)&0xFF); v.push_back(x&0xFF); }
    void be64(uint64_t x){ for(int i=7;i>=0;--i) v.push_back((x>>(8*i))&0xFF); }
    void bytes(const void* p, size_t n){ const uint8_t* b=(const uint8_t*)p; v.insert(v.end(), b, b+n); }
    void str4(const char s[4]){ v.push_back(s[0]); v.push_back(s[1]); v.push_back(s[2]); v.push_back(s[3]); }
};
struct BoxCtx{ size_t start; };
static BoxCtx box_begin(Buf& b, const char type[4]){ BoxCtx c{b.tell()}; b.be32(0); b.str4(type); return c; }
static void  box_end(Buf& b, BoxCtx c){ uint32_t sz=(uint32_t)(b.tell()-c.start); b.v[c.start+0]=(sz>>24)&0xFF; b.v[c.start+1]=(sz>>16)&0xFF; b.v[c.start+2]=(sz>>8)&0xFF; b.v[c.start+3]=sz&0xFF; }

static void write_matrix(Buf& b){
    b.be32(0x00010000); b.be32(0); b.be32(0);
    b.be32(0); b.be32(0x00010000); b.be32(0);
    b.be32(0); b.be32(0); b.be32(0x40000000);
}

// -------------------------- Bitreader (RBSP) --------------------------------
struct Br {
    const uint8_t* p; int nbits; int bitpos;
    Br(const uint8_t* data, int size_bytes): p(data), nbits(size_bytes*8), bitpos(0){}
    uint32_t get1(){ if(bitpos>=nbits) return 0; uint32_t v=(p[bitpos>>3]>>(7-(bitpos&7)))&1u; ++bitpos; return v; }
    uint32_t getn(int n){ uint32_t v=0; for(int i=0;i<n;++i) v=(v<<1)|get1(); return v; }
    uint32_t ue(){ int z=0; while(bitpos<nbits && get1()==0) ++z; uint32_t info = (z? getn(z):0); return ((1u<<z)-1u)+info; }
    int32_t  se(){ uint32_t u=ue(); int32_t s = (u&1)? (int32_t)((u+1)/2) : -(int32_t)(u/2); return s; }
};

// remove emulation prevention bytes (Annex B RBSP -> RBSP)
static std::vector<uint8_t> rbsp_from_nal(const uint8_t* nal, int sz, int header_bytes){
    std::vector<uint8_t> out; out.reserve(sz);
    for(int i=header_bytes;i<sz;++i){
        if (i+2<sz && nal[i]==0&&nal[i+1]==0&&nal[i+2]==3){ out.push_back(0); out.push_back(0); i+=2; continue; }
        out.push_back(nal[i]);
    }
    return out;
}

// ------------------ H.264 SPS -> width/height -------------------------------
static bool h264_parse_sps_wh(const uint8_t* sps, int sps_len, unsigned& W, unsigned& H){
    if (sps_len < 4) return false;
    // NAL header is 1 byte for AVC
    auto rb = rbsp_from_nal(sps, sps_len, 1);
    Br br(rb.data(), (int)rb.size());
    (void)br.getn(8); // profile_idc
    (void)br.getn(8); // constraint flags + reserved
    (void)br.getn(8); // level_idc
    (void)br.ue();    // seq_parameter_set_id
    // skip optional chroma_format_idc et al for high profiles (rough, safe):
    // We don't strictly need chroma for width/height parsing that follows.
    // But to be robust, check profile_idc from RBSP (first byte)
    uint8_t profile_idc = rb[0];
    if (profile_idc==100||profile_idc==110||profile_idc==122||profile_idc==244||profile_idc==44||profile_idc==83||profile_idc==86||profile_idc==118||profile_idc==128||profile_idc==138||profile_idc==139||profile_idc==134){
        uint32_t chroma_format_idc = br.ue();
        if (chroma_format_idc==3) (void)br.get1(); // separate_colour_plane_flag
        (void)br.ue(); // bit_depth_luma_minus1
        (void)br.ue(); // bit_depth_chroma_minus1
        (void)br.get1(); // qpprime_y_zero_transform_bypass_flag
        uint32_t seq_scaling_matrix_present_flag = br.get1();
        if (seq_scaling_matrix_present_flag){
            // skip scaling lists (compactly)
            int cnt = (chroma_format_idc==3)? 12:8;
            for(int i=0;i<cnt;++i){ if (br.get1()){ /*scaling_list*/ int last=8,next=8; int size = (i<6)?16:64; for(int j=0;j<size;++j){ int32_t v = br.se(); next=(last+v+256)%256; last=next; if (next==0){} } } }
        }
    }
    uint32_t log2_max_frame_num_minus4 = br.ue(); (void)log2_max_frame_num_minus4;
    uint32_t pic_order_cnt_type = br.ue();
    if (pic_order_cnt_type==0) (void)br.ue();
    else if (pic_order_cnt_type==1){ (void)br.get1(); (void)br.se(); (void)br.se(); uint32_t n=br.ue(); for(uint32_t i=0;i<n;++i) (void)br.se(); }
    (void)br.ue(); // max_num_ref_frames
    (void)br.get1(); // gaps_in_frame_num_value_allowed_flag
    uint32_t pic_width_in_mbs_minus1 = br.ue();
    uint32_t pic_height_in_map_units_minus1 = br.ue();
    uint32_t frame_mbs_only_flag = br.get1();
    if (!frame_mbs_only_flag) (void)br.get1(); // mb_adaptive_frame_field_flag
    (void)br.get1(); // direct_8x8_inference_flag
    uint32_t frame_cropping_flag = br.get1();
    uint32_t crop_left=0,crop_right=0,crop_top=0,crop_bottom=0;
    if (frame_cropping_flag){
        crop_left   = br.ue();
        crop_right  = br.ue();
        crop_top    = br.ue();
        crop_bottom = br.ue();
    }
    // vui_parameters_present_flag ignored for width/height
    unsigned width = (pic_width_in_mbs_minus1+1)*16;
    unsigned height= (pic_height_in_map_units_minus1+1)*16*(frame_mbs_only_flag?1:2);

    // crop units depend on chroma format; assume 4:2:0 defaults (typical)
    unsigned crop_unit_x=1, crop_unit_y=2; // for 4:2:0
    // That's good enough for auto-res in most material.
    width  -= (crop_left+crop_right)*crop_unit_x;
    height -= (crop_top +crop_bottom)*crop_unit_y;

    if (!width || !height) return false;
    W=width; H=height; return true;
}

// ------------------ H.265 SPS -> width/height -------------------------------
static void skip_profile_tier_level(Br& br, uint32_t max_sub_layers_minus1){
    // general_*
    (void)br.getn(2);   // general_profile_space
    (void)br.getn(1);   // general_tier_flag
    (void)br.getn(5);   // general_profile_idc
    (void)br.getn(32);  // general_profile_compatibility_flags
    (void)br.getn(48);  // general_constraint_indicator_flags
    (void)br.getn(8);   // general_level_idc

    // sub_layer_* present flags (length = max_sub_layers_minus1)
    std::vector<uint8_t> sub_layer_profile_present_flag(max_sub_layers_minus1, 0);
    std::vector<uint8_t> sub_layer_level_present_flag(max_sub_layers_minus1, 0);
    for (uint32_t i=0; i<max_sub_layers_minus1; ++i) {
        sub_layer_profile_present_flag[i] = br.get1();
        sub_layer_level_present_flag[i]   = br.get1();
    }

    // Only when max_sub_layers_minus1 > 0
    if (max_sub_layers_minus1 > 0) {
        for (uint32_t i = max_sub_layers_minus1; i < 8; ++i) {
            (void)br.getn(2); // reserved_zero_2bits
        }
    }

    for (uint32_t i=0; i<max_sub_layers_minus1; ++i) {
        if (sub_layer_profile_present_flag[i]) {
            (void)br.getn(2);
            (void)br.getn(1);
            (void)br.getn(5);
            (void)br.getn(32);
            (void)br.getn(48);
        }
        if (sub_layer_level_present_flag[i]) {
            (void)br.getn(8);
        }
    }
}

static bool h265_parse_sps_wh(const uint8_t* sps, int sps_len,
                              unsigned& W, unsigned& H, uint8_t& max_sub_layers_minus1_out){
    if (sps_len < 4) return false;

    // Strip emulation-prevention bytes, skip 2-byte NAL header
    auto rb = rbsp_from_nal(sps, sps_len, 2);
    Br br(rb.data(), (int)rb.size());

    (void)br.getn(4); // sps_video_parameter_set_id
    uint32_t sps_max_sub_layers_minus1 = br.getn(3);
    max_sub_layers_minus1_out = (uint8_t)sps_max_sub_layers_minus1;
    (void)br.get1();  // sps_temporal_id_nesting_flag

    // Properly skip profile_tier_level (this was the misalignment)
    skip_profile_tier_level(br, sps_max_sub_layers_minus1);

    (void)br.ue();                // sps_seq_parameter_set_id
    uint32_t chroma_format_idc = br.ue();
    if (chroma_format_idc == 3) (void)br.get1(); // separate_colour_plane_flag

    uint32_t pic_width_in_luma_samples  = br.ue();
    uint32_t pic_height_in_luma_samples = br.ue();

    uint32_t conformance_window_flag = br.get1();
    uint32_t conf_win_left=0, conf_win_right=0, conf_win_top=0, conf_win_bottom=0;
    if (conformance_window_flag){
        conf_win_left   = br.ue();
        conf_win_right  = br.ue();
        conf_win_top    = br.ue();
        conf_win_bottom = br.ue();
    }

    // Subsampling factors
    uint32_t subWidthC  = (chroma_format_idc == 1 || chroma_format_idc == 2) ? 2u : 1u;
    uint32_t subHeightC = (chroma_format_idc == 1) ? 2u : 1u;

    uint32_t width  = pic_width_in_luma_samples
                    - subWidthC  * (conf_win_right + conf_win_left);
    uint32_t height = pic_height_in_luma_samples
                    - subHeightC * (conf_win_top   + conf_win_bottom);

    if (!width || !height) return false;
    W = (unsigned)width; H = (unsigned)height;
    return true;
}

// -------------------------- MP4 boxes shared --------------------------------
static void write_ftyp(Buf& b, bool h265){
    BoxCtx c = box_begin(b, "ftyp");
    b.str4("isom"); b.be32(512);
    b.str4("isom"); b.str4("iso6"); b.str4("mp41"); b.str4(h265? "hvc1":"avc1");
    box_end(b,c);
}

// avcC builder: one SPS, one PPS (raw NAL bytes, no start code)
static void write_avcC(Buf& b, const std::vector<uint8_t>& sps, const std::vector<uint8_t>& pps){
    if (sps.size()<4 || pps.empty()) mp4w_die("avcC: missing SPS/PPS");
    BoxCtx c = box_begin(b, "avcC");
    b.u8(1);
    b.u8(sps[1]); b.u8(sps[2]); b.u8(sps[3]);  // profile/compat/level from SPS header bytes
    b.u8(0xFF); // lengthSizeMinusOne=3 (4-byte lengths)
    b.u8(0xE1); // numOfSPS=1
    b.be16((uint16_t)sps.size()); b.bytes(sps.data(), sps.size());
    b.u8(1); // numOfPPS=1
    b.be16((uint16_t)pps.size()); b.bytes(pps.data(), pps.size());
    box_end(b,c);
}

// Minimal hvcc builder: VPS/SPS/PPS arrays + profile fields borrowed from SPS ptl
static void write_hvcc(Buf& b,
                       const std::vector<uint8_t>& vps,
                       const std::vector<uint8_t>& sps,
                       const std::vector<uint8_t>& pps){
    if (sps.size()<6) mp4w_die("hvcc: missing SPS");
    BoxCtx c = box_begin(b, "hvcC");
    b.u8(1); // configurationVersion
    // Parse a few bits from SPS payload (copy ptl fields from rbsp quickly)
    // For simplicity, mirror common values: profile_space/tier/profile_idc/compat/constraint/level
    // We'll re-read from RBSP to fill these:
    uint8_t max_sub_layers_minus1_dummy=0;
    unsigned _W=0,_H=0; (void)_W; (void)_H;
    // Build rbsp to pick ptl bytes directly (after 2-byte NAL hdr)
    auto rb = rbsp_from_nal(sps.data(), (int)sps.size(), 2);
    Br br(rb.data(), (int)rb.size());
    (void)br.getn(4);
    uint32_t sps_max_sub_layers_minus1 = br.getn(3);
    (void)br.get1();
    // Now we need the raw bitfields; re-consume into temporary bytes for general_*:
    // To avoid complexity, set commonly-accepted defaults and level from typical SPS:
    // general_profile_space(2)=0, tier_flag(1)=0, profile_idc(5)=1 (Main) as fallback.
    uint8_t general_profile_space = 0;
    uint8_t general_tier_flag     = 0;
    uint8_t general_profile_idc   = 1; // Main
    uint32_t general_profile_compatibility_flags = 0;
    uint64_t general_constraint_indicator_flags  = 0;
    uint8_t general_level_idc = 120; // Level 4.0 (fallback)

    // A tiny attempt to read level_idc: skip to it quickly (see detailed parser above).
    // We'll just try to reinstantiate the ptl head with a second Br and consume fields.
    {
        Br br2(rb.data(), (int)rb.size());
        (void)br2.getn(4); (void)br2.getn(3); (void)br2.get1();
        general_profile_space = (uint8_t)br2.getn(2);
        general_tier_flag     = (uint8_t)br2.getn(1);
        general_profile_idc   = (uint8_t)br2.getn(5);
        general_profile_compatibility_flags = br2.getn(32);
        // constraint flags (48 bits)
        uint32_t hi = br2.getn(16);
        uint32_t lo = br2.getn(32);
        general_constraint_indicator_flags = ((uint64_t)hi<<32) | lo;
        general_level_idc = (uint8_t)br2.getn(8);
    }

    b.u8((uint8_t)((general_profile_space<<6) | (general_tier_flag<<5) | (general_profile_idc & 0x1F)));
    b.be32(general_profile_compatibility_flags);
    // 48-bit constraint flags:
    b.be24((uint32_t)(general_constraint_indicator_flags >> 24));
    b.be24((uint32_t)(general_constraint_indicator_flags & 0xFFFFFFu));
    b.u8(general_level_idc);

    // min_spatial_segmentation_idc + parallelism/chroma/bitdepth fields:
    b.be16(0xF000);           // reserved(12) + min_spatial_segmentation_idc(12)=0
    b.u8(0xFC | 0);           // reserved(6) + parallelismType(2)=0
    b.u8(0xFC | 1);           // reserved(6) + chromaFormat(2)=1 (4:2:0)
    b.u8(0xF8 | 0);           // reserved(5) + bitDepthLumaMinus8(3)=0 (8-bit)
    b.u8(0xF8 | 0);           // reserved(5) + bitDepthChromaMinus8(3)=0 (8-bit)
    b.be16(0);                // avgFrameRate=0 (we don't hard-code here)
    b.u8(0x00);               // constantFrameRate(2)=0, numTemporalLayers(3)=0, temporalIdNested(1)=0, lengthSizeMinusOne(2)=3-> set below
    b.v.back() |= 0x03;       // lengthSizeMinusOne = 3 (4-byte lengths)

    // arrays: VPS(32), SPS(33), PPS(34)
    auto write_array = [&](uint8_t nal_unit_type, const std::vector<uint8_t>& x){
        if (x.empty()) return;
        b.u8(0x80 | (nal_unit_type & 0x3F)); // array_completeness=1, reserved=0, nal_unit_type
        b.be16(1); // numNalus
        b.be16((uint16_t)x.size());
        b.bytes(x.data(), x.size());
    };
    int arrays = 0; if(!vps.empty()) ++arrays; if(!sps.empty()) ++arrays; if(!pps.empty()) ++arrays;
    // We need numOfArrays before writing them — so cache pos
    // We'll temporarily write a placeholder and patch it:
    // After the "lengthSizeMinusOne" byte, we should write numOfArrays:
    // Insert here:
    // (We didn't write it yet; so push then patch.)
    // -> Let's rebuild: we will store position just before arrays start.
    // Easiest: Record current index, insert a dummy, then patch:
    size_t arrays_count_pos = b.v.size();
    b.u8(0); // placeholder for numOfArrays
    uint8_t count=0;
    if(!vps.empty()){ ++count; write_array(32, vps); }
    if(!sps.empty()){ ++count; write_array(33, sps); }
    if(!pps.empty()){ ++count; write_array(34, pps); }
    b.v[arrays_count_pos] = count;

    box_end(b,c);
}

static void write_moov(Buf& b, bool h265, unsigned width, unsigned height, uint32_t timescale, uint64_t duration90k,
                       const std::vector<uint8_t>& vps, const std::vector<uint8_t>& sps, const std::vector<uint8_t>& pps){
    BoxCtx moov = box_begin(b, "moov");
    // mvhd
    {
        BoxCtx c=box_begin(b, "mvhd");
        b.u8(0); b.be24(0);
        b.be32(0); b.be32(0);
        b.be32(timescale);
        b.be32((uint32_t)duration90k);
        b.be32(0x00010000);
        b.be16(0x0100); b.be16(0); b.be32(0); b.be32(0);
        write_matrix(b);
        for(int i=0;i<6;++i) b.be32(0);
        b.be32(1); // next_track_ID
        box_end(b,c);
    }

    // trak
    BoxCtx trak = box_begin(b, "trak");
    {
        // tkhd
        BoxCtx c=box_begin(b, "tkhd");
        b.u8(0); b.be24(0x0007);
        b.be32(0); b.be32(0);
        b.be32(1);
        b.be32(0);
        b.be32((uint32_t)duration90k);
        b.be32(0); b.be32(0);
        b.be16(0); b.be16(0); b.be16(0); b.be16(0);
        write_matrix(b);
        b.be32(width<<16); b.be32(height<<16);
        box_end(b,c);
    }
    // mdia
    BoxCtx mdia = box_begin(b, "mdia");
    {
        BoxCtx c=box_begin(b, "mdhd");
        b.u8(0); b.be24(0);
        b.be32(0); b.be32(0);
        b.be32(timescale);
        b.be32((uint32_t)duration90k);
        b.be16(0x55C4); b.be16(0);
        box_end(b,c);
    }
    {
        BoxCtx c=box_begin(b, "hdlr");
        b.u8(0); b.be24(0);
        b.be32(0);
        b.str4("vide");
        b.be32(0); b.be32(0); b.be32(0);
        const char name[]="VideoHandler\0"; b.bytes(name,sizeof(name));
        box_end(b,c);
    }
    // minf
    BoxCtx minf = box_begin(b, "minf");
    {
        BoxCtx c=box_begin(b, "vmhd"); b.u8(0); b.be24(1); b.be16(0); b.be16(0); b.be16(0); box_end(b,c);
    }
    {
        BoxCtx dinf=box_begin(b, "dinf");
        BoxCtx dref=box_begin(b, "dref"); b.u8(0); b.be24(0); b.be32(1);
        BoxCtx url = box_begin(b, "url "); b.u8(0); b.be24(1); box_end(b,url);
        box_end(b,dref); box_end(b,dinf);
    }
    // stbl
    BoxCtx stbl = box_begin(b, "stbl");
    {
        BoxCtx stsd=box_begin(b, "stsd"); b.u8(0); b.be24(0); b.be32(1);
        // sample entry
        BoxCtx se=box_begin(b, h265? "hvc1":"avc1");
        for(int i=0;i<6;++i) b.u8(0);
        b.be16(1);
        b.be16(0); b.be16(0); b.be32(0); b.be32(0); b.be32(0);
        b.be16((uint16_t)width); b.be16((uint16_t)height);
        b.be32(0x00480000); b.be32(0x00480000); b.be32(0);
        b.be16(1);
        { uint8_t name_len=0; b.u8(name_len); for(int i=0;i<31;++i) b.u8(0); }
        b.be16(0x0018); b.be16(0xFFFF);
        if (h265) write_hvcc(b, vps, sps, pps);
        else      write_avcC(b, sps, pps);
        box_end(b,se);
        box_end(b,stsd);
    }
    // empty stts/stsc/stsz/stco for fragmented
    { BoxCtx c=box_begin(b,"stts"); b.u8(0); b.be24(0); b.be32(0); box_end(b,c); }
    { BoxCtx c=box_begin(b,"stsc"); b.u8(0); b.be24(0); b.be32(0); box_end(b,c); }
    { BoxCtx c=box_begin(b,"stsz"); b.u8(0); b.be24(0); b.be32(0); b.be32(0); box_end(b,c); }
    { BoxCtx c=box_begin(b,"stco"); b.u8(0); b.be24(0); b.be32(0); box_end(b,c); }

    box_end(b,stbl);
    box_end(b,minf);
    box_end(b,mdia);
    box_end(b,trak);

    // mvex/trex (defaults mostly unused since we carry per-sample duration/size)
    {
        BoxCtx mvex=box_begin(b,"mvex");
        BoxCtx trex=box_begin(b,"trex");
        b.u8(0); b.be24(0);
        b.be32(1); b.be32(1); b.be32(0); b.be32(0); b.be32(0);
        box_end(b,trex);
        box_end(b,mvex);
    }

    box_end(b,moov);
}

// -------------------------- Fragment writer ---------------------------------
struct Sample {
    std::vector<uint8_t> data; // frame payload: concatenated length-prefixed NALUs (4-byte lengths)
    uint32_t duration90k = 0;  // per-sample duration in 90kHz ticks
    bool key = false;
};

static const uint32_t FLAGS_SYNC     = 0x02000000;
static const uint32_t FLAGS_NON_SYNC = 0x00010000;

static void write_fragment(FILE* fp,
                           uint32_t seqno,
                           uint64_t base_decode_time90k,
                           const std::vector<Sample>& samples){
    if (samples.empty()) return;

    // Build moof
    Buf m;
    BoxCtx moof = box_begin(m, "moof");
    {
        // mfhd
        { BoxCtx c=box_begin(m,"mfhd"); m.u8(0); m.be24(0); m.be32(seqno); box_end(m,c); }
        // traf
        BoxCtx traf=box_begin(m,"traf");
        // tfhd: default-base-is-moof
        {
            BoxCtx c=box_begin(m,"tfhd");
            m.u8(0); m.be24(0x020000); // default-base-is-moof
            m.be32(1); // track_ID
            box_end(m,c);
        }
        // tfdt (v1 for 64-bit)
        {
            BoxCtx c=box_begin(m,"tfdt");
            m.u8(1); m.be24(0);
            m.be64(base_decode_time90k);
            box_end(m,c);
        }
        // trun: we provide per-sample duration + size + flags + data-offset
        {
            BoxCtx c=box_begin(m,"trun");
            uint32_t flags = 0x000001 /*data offset*/ | 0x000100 /*duration*/ | 0x000200 /*size*/ | 0x000400 /*flags*/;
            m.u8(0); m.be24(flags);
            m.be32((uint32_t)samples.size());
            size_t data_offset_pos = m.tell(); m.be32(0);
            for (auto& s: samples){
                m.be32(s.duration90k);
                m.be32((uint32_t)s.data.size());
                m.be32(s.key ? FLAGS_SYNC : FLAGS_NON_SYNC);
            }
            box_end(m,c);
            uint32_t moof_size = (uint32_t)m.tell();
            uint32_t offset = moof_size + 8; // mdat header size
            m.v[data_offset_pos+0]=(offset>>24)&0xFF; m.v[data_offset_pos+1]=(offset>>16)&0xFF;
            m.v[data_offset_pos+2]=(offset>>8)&0xFF;  m.v[data_offset_pos+3]=(offset)&0xFF;
        }
        box_end(m,traf);
    }
    box_end(m,moof);

    // mdat
    uint64_t payload=0; for (auto& s: samples) payload += s.data.size();
    if (payload + 8 > 0xFFFFFFFFull) mp4w_die("mdat too big");
    uint32_t mdat32 = (uint32_t)(payload + 8);

    // write
    fwrite(m.v.data(), 1, m.v.size(), fp);
    uint8_t hdr[8] = { (uint8_t)(mdat32>>24), (uint8_t)(mdat32>>16), (uint8_t)(mdat32>>8), (uint8_t)(mdat32),
                       'm','d','a','t' };
    fwrite(hdr, 1, 8, fp);
    for (auto& s: samples) fwrite(s.data.data(), 1, s.data.size(), fp);
}

// -------------------------- NAL helpers -------------------------------------
static inline int h264_nal_type(const uint8_t* p){ return p[0] & 0x1F; }
static inline int h265_nal_type(const uint8_t* p){ return (p[0] >> 1) & 0x3F; }

static bool is_key_h264(const uint8_t* nal, int len){
    if (len<1) return false;
    int t = h264_nal_type(nal);
    return t==5; // IDR
}
static bool is_key_h265_type(int t){
    // IDR_W_RADL(19), IDR_N_LP(20), CRA(21), BLA* (16..18) are all random-access points
    return (t>=16 && t<=21);
}
static bool is_vps_sps_pps_h265_type(int t){ return t==32||t==33||t==34; }

// -------------------------- mp4_writer impl ---------------------------------
struct mp4_writer {
    std::string filename;
    bool is_h265=false;
    FILE* fp=nullptr;

    // codec config
    std::vector<uint8_t> vps, sps, pps;
    bool have_init=false;
    unsigned width=0, height=0;

    // timing
    static constexpr uint32_t TIMESCALE = 90000; // movie/track timescale
    uint64_t first_ts=0, last_ts=0;
    bool have_first_ts=false;

    // sample staging (we keep one-sample lag to know duration)
    std::vector<Sample> pending; // ready-to-flush samples in current fragment
    std::vector<uint8_t> prev_frame_data; // to compute duration for previous frame
    bool prev_key=false;
    uint64_t prev_ts=0;
    uint64_t sum_delta=0; uint64_t cnt_delta=0;

    uint32_t seqno=1; // fragment sequence

    uint64_t dts_accum90k = 0;        // sum of durations already flushed (fragment base)
    uint64_t pending_base90k = 0;     // DTS of first sample currently pending
    bool     have_pending_base = false;

    // fragmenting: start new fragment when encountering a keyframe (after the first)
    // We hold samples in 'pending' until next keyframe triggers flush.
};

// Write init when we know {width,height} + codec configs
static void maybe_write_init(mp4_writer* w, uint64_t provisional_duration=0){
    if (w->have_init) return;
    if (!w->width || !w->height) return;
    if (w->is_h265){
        if (w->sps.empty() || w->pps.empty()) return; // VPS optional but recommended
    } else {
        if (w->sps.empty() || w->pps.empty()) return;
    }
    Buf init;
    write_ftyp(init, w->is_h265);
    write_moov(init, w->is_h265, w->width, w->height, w->TIMESCALE, provisional_duration,
               w->vps, w->sps, w->pps);
    w->fp = std::fopen(w->filename.c_str(), "wb");
    if (!w->fp) mp4w_die("open output failed");
    fwrite(init.v.data(), 1, init.v.size(), w->fp);
    w->have_init = true;
}

// Flush current fragment (pending samples) starting at given base decode time
static void flush_fragment(mp4_writer* w, uint64_t base_time90k){
    if (!w->have_init || w->pending.empty()) return;
    write_fragment(w->fp, w->seqno++, base_time90k, w->pending);
    w->pending.clear();
}

// parse NALU array (length-prefixed) once to collect VPS/SPS/PPS and decide keyframe
static void scan_and_collect(mp4_writer* w, const uint8_t* data, int data_length,
                             bool& is_key, bool& saw_config){
    is_key=false; saw_config=false;
    int off=0;
    while (off + 4 <= data_length){
        uint32_t L = (data[off]<<24)|(data[off+1]<<16)|(data[off+2]<<8)|data[off+3];
        off += 4;
        if (off + (int)L > data_length) break;
        const uint8_t* nal = data+off; int nlen=(int)L;

        if (!w->is_h265){
            int t = h264_nal_type(nal);
            if (t==7 && w->sps.empty()){ w->sps.assign(nal, nal+nlen); saw_config=true; }
            else if (t==8 && w->pps.empty()){ w->pps.assign(nal, nal+nlen); saw_config=true; }
            else if (t==5) is_key=true;
        } else {
            int t = h265_nal_type(nal);
            if (t==32 && w->vps.empty()){ w->vps.assign(nal, nal+nlen); saw_config=true; }
            else if (t==33 && w->sps.empty()){ w->sps.assign(nal, nal+nlen); saw_config=true; }
            else if (t==34 && w->pps.empty()){ w->pps.assign(nal, nal+nlen); saw_config=true; }
            if (is_key_h265_type(t)) is_key=true;
        }
        off += nlen;
    }
}

// Detect resolution from SPS once available
static void maybe_parse_resolution(mp4_writer* w){
    if (w->width && w->height) return;
    unsigned W=0,H=0;
    if (!w->is_h265){
        if (!w->sps.empty()){
            if (h264_parse_sps_wh(w->sps.data(), (int)w->sps.size(), W, H)){ w->width=W; w->height=H; }
        }
    } else {
        if (!w->sps.empty()){
            uint8_t tmp=0;
            if (h265_parse_sps_wh(w->sps.data(), (int)w->sps.size(), W, H, tmp)){ w->width=W; w->height=H; }
        }
    }
}

mp4_writer_t *mp4_writer_create(const char *filename, bool h265){
    try{
        auto* w = new mp4_writer;
        w->filename = filename? filename : "";
        if (w->filename.empty()) mp4w_die("filename empty");
        w->is_h265 = h265;
        return w;
    }catch(...){ return nullptr; }
}


// data = frame payload as concatenated [4-byte len][nalu]... (AVCC/HVCC)
// extended_90khz_timestamp = absolute 90kHz ticks (monotonic, non-decreasing)
void mp4_writer_add_video_frame(mp4_writer_t *wr,
                                uint64_t extended_90khz_timestamp,
                                uint8_t *data, int data_length){
    if (!wr || !data || data_length<=0) return;
    auto* w = (mp4_writer*)wr;

    // Collect VPS/SPS/PPS and keyframe flag from this frame
    bool is_key=false, saw_cfg=false;
    scan_and_collect(w, data, data_length, is_key, saw_cfg);

    // If we just learned SPS(/VPS/PPS), try to parse resolution & maybe write init
    if (saw_cfg){
        maybe_parse_resolution(w);
        if (!w->have_init) maybe_write_init(w, 0);
    }

    // Timing: compute duration for the *previous* frame once we see this ts.
    if (!w->have_first_ts){
        w->first_ts = extended_90khz_timestamp;
        w->prev_ts  = extended_90khz_timestamp;
        w->have_first_ts = true;
        // Stash the first frame's payload; its duration will be known at next call
        w->prev_frame_data.assign(data, data+data_length);
        w->prev_key = is_key;
        return;
    }

    uint64_t delta = (extended_90khz_timestamp >= w->prev_ts)
                   ? (extended_90khz_timestamp - w->prev_ts)
                   : 0;
    if (delta==0) delta = 1; // avoid zero duration

    // The *previous* frame can now be finalized with known duration
    Sample s;
    s.data = std::move(w->prev_frame_data);
    s.duration90k = (uint32_t)std::min<uint64_t>(delta, std::numeric_limits<uint32_t>::max());
    s.key = w->prev_key;
    // Starting a new fragment on keyframes (except very first sample ever)
    bool should_flush = (!w->pending.empty() && is_key);
    // Base time of the current fragment == DTS of first sample in pending
    uint64_t base_time = 0;
    if (w->pending.empty()){
        base_time = w->prev_ts; // DTS of the first sample we'll push
    } else {
        // DTS of fragment start = first sample's DTS = base of pending
        base_time = w->last_ts; // last flushed base + accumulated? We'll set correctly below.
        // We actually track base at flush time (see below).
    }
    if (!w->have_pending_base) {
        w->pending_base90k = w->dts_accum90k;  // first pending sample starts at current accumulated DTS
        w->have_pending_base = true;
    }
    // Push finalized previous sample
    w->pending.push_back(std::move(s));
    w->sum_delta += delta; w->cnt_delta += 1;

    auto flush_now = [&](mp4_writer* w){
        if (!w->have_init || w->pending.empty()) return;
        // compute total duration of the pending samples
        uint64_t frag_dur = 0;
        for (auto& x : w->pending) frag_dur += x.duration90k;

        write_fragment(w->fp, w->seqno++, w->pending_base90k, w->pending);

        w->dts_accum90k += frag_dur;
        w->pending.clear();
        w->have_pending_base = false;
    };

    // If we have init and a keyframe boundary, flush the fragment up to now.
    if (is_key && !w->pending.empty()) {
        flush_now(w);
    }

    // Update last_ts
    w->last_ts = w->prev_ts; // the sample we just finalized started at prev_ts
    // Start a new fragment on the soon-to-be-added current (key) frame:
    if (should_flush && w->have_init){
        // Base time for the pending fragment is the DTS of its first sample.
        // That DTS equals (last flush time), which we can compute as:
        // w->first_ts + sum of durations of all samples written so far.
        // To avoid tracking a separate accumulator, compute it on the fly:
        static uint64_t written_time = 0; // function-static shared across writers would be wrong; so:
        // Instead, store it per-writer (add a field). Let's do that:
        // (Retrofit now)
    }
    // We realize we need per-writer "accumulated_time90k" of all flushed samples:
    w->prev_frame_data.assign(data, data + data_length);
    w->prev_key = is_key;
    w->prev_ts  = extended_90khz_timestamp;
}

void mp4_writer_destroy(mp4_writer_t *wr){
    if (!wr) return;
    auto* w = (mp4_writer*)wr;
    try{
        // Finalize: we still have one "prev_frame_data" not yet emitted (no next ts)
        if (w->have_first_ts && !w->prev_frame_data.empty()){
            uint32_t avg = w->cnt_delta ? (uint32_t)std::max<uint64_t>(1, w->sum_delta / w->cnt_delta) : 3003;
            if (!w->have_pending_base) { w->pending_base90k = w->dts_accum90k; w->have_pending_base = true; }
            Sample s; s.data = std::move(w->prev_frame_data); s.duration90k = avg; s.key = w->prev_key;
            w->pending.push_back(std::move(s));
        }

        // make sure init is written
        if (!w->have_init){ maybe_parse_resolution(w); maybe_write_init(w, 0); }

        // flush remaining samples with base = pending_base90k
        if (w->have_init && !w->pending.empty()){
            uint64_t frag_dur = 0; for (auto& x : w->pending) frag_dur += x.duration90k;
            write_fragment(w->fp, w->seqno++, w->pending_base90k, w->pending);
            w->dts_accum90k += frag_dur;
            w->pending.clear();
            w->have_pending_base = false;
        }

        if (w->fp){ std::fclose(w->fp); w->fp=nullptr; }
    }catch(...){
        if (w->fp){ std::fclose(w->fp); w->fp=nullptr; }
    }
    delete w;
}
