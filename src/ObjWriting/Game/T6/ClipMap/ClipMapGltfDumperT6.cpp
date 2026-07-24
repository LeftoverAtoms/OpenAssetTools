#include "ClipMapGltfDumperT6.h"

#include "Utils/Alignment.h"
#include "Utils/Logging/Log.h"
#include "XModel/Gltf/JsonGltf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace T6;

namespace
{
    constexpr auto WRITE_DEBUG_FILE = false;

    constexpr auto EPSILON = 0.125f;
    constexpr auto MAX_BRUSHES_PER_INLINE_MODEL = 256u;

    constexpr auto MERGE_COLLISION_TYPES = true;

    struct MeshPrimitive
    {
        std::string name;
        std::string groupName;
        std::vector<vec3_t> vertices;
        std::vector<vec2_t> uvs;
        std::vector<uint32_t> indices;
        vec3_t color;
    };

    struct MaterialTextureNames
    {
        std::string colorMapName;
    };

    struct ParsedMapEntity
    {
        std::map<std::string, std::string> values;
    };

    struct InlineModelCollisionSet
    {
        unsigned modelIndex;
        const ClipInfo* clipInfo;
        vec3_t origin;
        vec3_t angles;
        std::vector<unsigned> brushIndices;
        std::vector<int> aabbTreeIndices;
        std::vector<int> partitionIndices;
        std::string source;
    };

    struct InlineModelTransform
    {
        vec3_t origin;
        vec3_t angles;
        std::map<std::string, std::string> values;
    };

    struct BrushModelBrushes
    {
        const ClipInfo* clipInfo;
        std::vector<unsigned> indices;
        std::string source;
        bool usesOwnClipInfo;
    };

    using LeafBrushNodeCache = std::map<int, std::vector<unsigned>>;

    template<typename T> void SortUnique(std::vector<T>& values)
    {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    }

    vec3_t operator+(const vec3_t& a, const vec3_t& b)
    {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    vec3_t operator-(const vec3_t& a, const vec3_t& b)
    {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    vec3_t operator/(const vec3_t& a, const float s)
    {
        return {a.x / s, a.y / s, a.z / s};
    }

    float Dot(const vec3_t& a, const vec3_t& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    vec3_t Cross(const vec3_t& a, const vec3_t& b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        };
    }

    float Length(const vec3_t& v)
    {
        return std::sqrt(Dot(v, v));
    }

    vec3_t Normalize(const vec3_t& v)
    {
        const auto length = Length(v);
        if (length <= std::numeric_limits<float>::epsilon())
            return {0.0f, 0.0f, 1.0f};

        return v / length;
    }

    bool IsZero(const vec3_t& v)
    {
        return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
    }

    void AnglesToAxis(const vec3_t& angles, std::array<vec3_t, 3>& axis)
    {
        constexpr auto DEG_TO_RAD = 3.14159265358979323846f / 180.0f;

        const auto pitch = angles.x * DEG_TO_RAD;
        const auto yaw = angles.y * DEG_TO_RAD;
        const auto roll = angles.z * DEG_TO_RAD;

        const auto sp = std::sin(pitch);
        const auto cp = std::cos(pitch);
        const auto sy = std::sin(yaw);
        const auto cy = std::cos(yaw);
        const auto sr = std::sin(roll);
        const auto cr = std::cos(roll);

        axis[0] = {cp * cy, cp * sy, -sp};
        axis[1] = {sr * sp * cy + cr * -sy, sr * sp * sy + cr * cy, sr * cp};
        axis[2] = {cr * sp * cy + -sr * -sy, cr * sp * sy + -sr * cy, cr * cp};
    }

    vec3_t RotateByAxis(const vec3_t& point, const std::array<vec3_t, 3>& axis)
    {
        return {
            point.x * axis[0].x + point.y * axis[1].x + point.z * axis[2].x,
            point.x * axis[0].y + point.y * axis[1].y + point.z * axis[2].y,
            point.x * axis[0].z + point.y * axis[1].z + point.z * axis[2].z,
        };
    }

    vec3_t TransformPoint(const vec3_t& point, const InlineModelTransform& transform)
    {
        if (IsZero(transform.angles))
            return point + transform.origin;

        // CM_TransformedBoxTraceRotated @ .text:006AE660 moves traces into model
        // space by dot(start - origin, axis[i]) and transforms results back as
        // origin + axis[0] * x + axis[1] * y + axis[2] * z. Export uses that
        // same local-to-world direction for inline model brushes and triangles.
        std::array<vec3_t, 3> axis{};
        AnglesToAxis(transform.angles, axis);
        return RotateByAxis(point, axis) + transform.origin;
    }

    std::string Hex(const uint32_t value)
    {
        std::ostringstream ss;
        ss << std::hex << std::uppercase << value;
        return ss.str();
    }

    vec3_t ColorForIndex(const unsigned index)
    {
        constexpr vec3_t COLORS[]{
            {0.90f, 0.15f, 0.12f},
            {0.10f, 0.62f, 0.95f},
            {0.18f, 0.72f, 0.28f},
            {0.95f, 0.68f, 0.12f},
            {0.70f, 0.28f, 0.92f},
            {0.05f, 0.78f, 0.70f},
            {0.95f, 0.35f, 0.55f},
            {0.55f, 0.55f, 0.55f},
        };

        return COLORS[index % std::size(COLORS)];
    }

    bool NearlyEqual(const float a, const float b)
    {
        return std::fabs(a - b) <= EPSILON;
    }

    vec3_t PickPerpendicular(const vec3_t& normal)
    {
        const auto up = std::fabs(normal.z) < 0.9f ? vec3_t{{0.0f, 0.0f, 1.0f}} : vec3_t{{0.0f, 1.0f, 0.0f}};
        return Normalize(Cross(up, normal));
    }

    vec2_t PlanarUvForVertex(const vec3_t& vertex, const vec3_t& normal)
    {
        constexpr auto TEXTURE_SCALE = 1.0f / 64.0f;
        const auto flipUvVertically = [](const vec2_t uv)
        {
            return vec2_t{uv.x, -uv.y};
        };

        const auto absX = std::fabs(normal.x);
        const auto absY = std::fabs(normal.y);
        const auto absZ = std::fabs(normal.z);

        if (absZ >= absX && absZ >= absY)
            return flipUvVertically({vertex.x * TEXTURE_SCALE, (normal.z >= 0.0f ? vertex.y : -vertex.y) * TEXTURE_SCALE});

        if (absX >= absY)
            return flipUvVertically({(normal.x >= 0.0f ? vertex.y : -vertex.y) * TEXTURE_SCALE, vertex.z * TEXTURE_SCALE});

        return flipUvVertically({(normal.y >= 0.0f ? -vertex.x : vertex.x) * TEXTURE_SCALE, vertex.z * TEXTURE_SCALE});
    }

    vec3_t TriangleNormal(const vec3_t& a, const vec3_t& b, const vec3_t& c)
    {
        return Normalize(Cross(b - a, c - a));
    }

    void SortFaceVertices(std::vector<uint32_t>& faceIndices, const std::vector<vec3_t>& vertices, const vec3_t& normal)
    {
        vec3_t center{};
        for (const auto index : faceIndices)
            center = center + vertices[index];

        center = center / static_cast<float>(faceIndices.size());

        const auto tangent = PickPerpendicular(normal);
        const auto bitangent = Cross(normal, tangent);

        std::sort(faceIndices.begin(), faceIndices.end(),
                  [&](const uint32_t a, const uint32_t b)
                  {
                      const auto va = vertices[a] - center;
                      const auto vb = vertices[b] - center;
                      const auto angleA = std::atan2(Dot(va, bitangent), Dot(va, tangent));
                      const auto angleB = std::atan2(Dot(vb, bitangent), Dot(vb, tangent));
                      return angleA < angleB;
                  });
    }

    void AppendBytes(std::vector<uint8_t>& buffer, const void* data, const size_t size)
    {
        const auto* bytes = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), bytes, bytes + size);
    }

    void PadBuffer(std::vector<uint8_t>& buffer, const size_t alignment)
    {
        const auto alignedSize = utils::Align(buffer.size(), alignment);
        buffer.resize(alignedSize);
    }

    cplane_s MakePlane(const vec3_t& normal, const float dist)
    {
        return {normal, dist, 0, 0, {}};
    }

    cbrushside_t MakeBrushSide(cplane_s* plane, const int contentFlags, const int surfaceFlags)
    {
        return {plane, contentFlags, surfaceFlags};
    }

    std::vector<cbrushside_t> BuildBrushSides(const cbrush_t& brush, std::array<cplane_s, 6>& axialPlanes)
    {
        axialPlanes = {
            MakePlane({{1.0f, 0.0f, 0.0f}}, brush.maxs.x),
            MakePlane({{0.0f, 1.0f, 0.0f}}, brush.maxs.y),
            MakePlane({{0.0f, 0.0f, 1.0f}}, brush.maxs.z),
            MakePlane({{-1.0f, 0.0f, 0.0f}}, -brush.mins.x),
            MakePlane({{0.0f, -1.0f, 0.0f}}, -brush.mins.y),
            MakePlane({{0.0f, 0.0f, -1.0f}}, -brush.mins.z),
        };

        std::vector<cbrushside_t> sides;

        // CMod_LoadBrushes @ .text:0069CF34..0069D039 stores mins materials in
        // axial_*flags[0][axis] and maxs materials in axial_*flags[1][axis].
        sides.emplace_back(MakeBrushSide(&axialPlanes[0], brush.axial_cflags[1][0], brush.axial_sflags[1][0]));
        sides.emplace_back(MakeBrushSide(&axialPlanes[1], brush.axial_cflags[1][1], brush.axial_sflags[1][1]));
        sides.emplace_back(MakeBrushSide(&axialPlanes[2], brush.axial_cflags[1][2], brush.axial_sflags[1][2]));
        sides.emplace_back(MakeBrushSide(&axialPlanes[3], brush.axial_cflags[0][0], brush.axial_sflags[0][0]));
        sides.emplace_back(MakeBrushSide(&axialPlanes[4], brush.axial_cflags[0][1], brush.axial_sflags[0][1]));
        sides.emplace_back(MakeBrushSide(&axialPlanes[5], brush.axial_cflags[0][2], brush.axial_sflags[0][2]));

        for (auto sideIndex = 0u; brush.sides != nullptr && sideIndex < brush.numsides; sideIndex++)
        {
            const auto* plane = brush.sides[sideIndex].plane;
            if (plane != nullptr)
                sides.emplace_back(brush.sides[sideIndex]);
        }

        return sides;
    }

    void AppendTriangle(MeshPrimitive& mesh, const vec3_t& a, const vec3_t& b, const vec3_t& c)
    {
        const auto baseIndex = static_cast<uint32_t>(mesh.vertices.size());
        const auto normal = TriangleNormal(a, b, c);
        mesh.vertices.emplace_back(a);
        mesh.vertices.emplace_back(b);
        mesh.vertices.emplace_back(c);
        mesh.uvs.emplace_back(PlanarUvForVertex(a, normal));
        mesh.uvs.emplace_back(PlanarUvForVertex(b, normal));
        mesh.uvs.emplace_back(PlanarUvForVertex(c, normal));
        mesh.indices.emplace_back(baseIndex);
        mesh.indices.emplace_back(baseIndex + 1u);
        mesh.indices.emplace_back(baseIndex + 2u);
    }

    bool IsExcludedIndex(const int index, const std::vector<int>& excludedIndices)
    {
        return std::binary_search(excludedIndices.begin(), excludedIndices.end(), index);
    }

    void CollectCollisionPartitionsRecursive(std::vector<int>& result, const clipMap_t& clipMap, const int treeIndex)
    {
        if (treeIndex < 0 || treeIndex >= clipMap.aabbTreeCount)
            return;

        const auto& tree = clipMap.aabbTrees[treeIndex];
        if (tree.childCount != 0u)
        {
            // CM_TraceThroughAabbTree_r @ .text:006A1500 treats non-zero
            // childCount as an internal node and walks cm.aabbTrees[firstChildIndex + i].
            for (auto childOffset = 0u; childOffset < tree.childCount; childOffset++)
                CollectCollisionPartitionsRecursive(result, clipMap, tree.u.firstChildIndex + static_cast<int>(childOffset));

            return;
        }

        // CM_TraceThroughAabbTree_work @ .text:006A1430 reads the leaf tree's
        // u.partitionIndex from +0x1C and traces cm.partitions[partitionIndex].
        const auto partitionIndex = tree.u.partitionIndex;
        if (partitionIndex >= 0 && partitionIndex < clipMap.partitionCount)
            result.emplace_back(partitionIndex);
    }

    std::vector<int> CollectCollisionPartitionIndicesForRoots(const clipMap_t& clipMap, const std::vector<int>& rootTreeIndices)
    {
        std::vector<int> partitionIndices;
        for (const auto rootTreeIndex : rootTreeIndices)
            CollectCollisionPartitionsRecursive(partitionIndices, clipMap, rootTreeIndex);

        SortUnique(partitionIndices);
        return partitionIndices;
    }

    MaterialTextureNames ExtractMaterialTextureNames(const Material& material)
    {
        MaterialTextureNames textureNames;
        if (material.textureTable == nullptr)
            return textureNames;

        for (auto textureIndex = 0u; textureIndex < material.textureCount; textureIndex++)
        {
            const auto& textureDef = material.textureTable[textureIndex];
            if (textureDef.image == nullptr || textureDef.image->name == nullptr || textureDef.image->name[0] == '\0')
                continue;

            switch (textureDef.semantic)
            {
            case TS_COLOR_MAP:
            case TS_COLOR0_MAP:
            case TS_COLOR1_MAP:
            case TS_COLOR2_MAP:
            case TS_COLOR3_MAP:
            case TS_COLOR4_MAP:
            case TS_COLOR5_MAP:
            case TS_COLOR6_MAP:
            case TS_COLOR7_MAP:
            case TS_COLOR8_MAP:
            case TS_COLOR9_MAP:
            case TS_COLOR10_MAP:
            case TS_COLOR11_MAP:
            case TS_COLOR12_MAP:
            case TS_COLOR13_MAP:
            case TS_COLOR14_MAP:
            case TS_COLOR15_MAP:
                if (textureNames.colorMapName.empty())
                    textureNames.colorMapName = textureDef.image->name;
                break;

            default:
                break;
            }
        }

        return textureNames;
    }

    std::string MaterialBaseName(const std::string_view materialName)
    {
        const auto lastSlash = materialName.find_last_of("/\\");
        if (lastSlash == std::string_view::npos)
            return std::string(materialName);

        return std::string(materialName.substr(lastSlash + 1u));
    }

    bool IsPreferredWorldMaterialName(const std::string_view materialName)
    {
        return materialName.starts_with("wpc/") || materialName.starts_with("wpc\\");
    }

    std::map<std::string, MaterialTextureNames> BuildMaterialTextureLookup(const AssetDumpingContext& context)
    {
        std::map<std::string, MaterialTextureNames> lookup;
        for (const auto* materialAsset : context.m_zone.m_pools.PoolAssets<AssetMaterial>())
        {
            const auto* material = materialAsset->Asset();
            if (material == nullptr || material->info.name == nullptr || material->info.name[0] == '\0')
                continue;

            auto textureNames = ExtractMaterialTextureNames(*material);
            if (textureNames.colorMapName.empty())
                continue;

            const auto fullMaterialName = std::string(material->info.name);
            lookup.emplace(fullMaterialName, textureNames);

            const auto baseMaterialName = MaterialBaseName(fullMaterialName);
            const auto existingBaseName = lookup.find(baseMaterialName);
            if (existingBaseName == lookup.end() || IsPreferredWorldMaterialName(fullMaterialName))
                lookup[baseMaterialName] = std::move(textureNames);
        }

        return lookup;
    }

    unsigned ClaimCollisionPartitionsForModel(std::vector<int>& partitionIndices, std::vector<int>& partitionModelOwner, const unsigned modelIndex)
    {
        auto duplicatePartitions = 0u;
        for (auto& partitionIndex : partitionIndices)
        {
            if (partitionIndex < 0 || partitionIndex >= static_cast<int>(partitionModelOwner.size()))
                continue;

            if (partitionModelOwner[partitionIndex] < 0)
            {
                partitionModelOwner[partitionIndex] = static_cast<int>(modelIndex);
                continue;
            }

            partitionIndex = -1;
            duplicatePartitions++;
        }

        partitionIndices.erase(std::remove(partitionIndices.begin(), partitionIndices.end(), -1), partitionIndices.end());
        return duplicatePartitions;
    }

    void AppendCollisionTreeTriangles(
        std::map<std::string, MeshPrimitive>& byMaterial,
        std::vector<bool>& emittedTriangles,
        const clipMap_t& clipMap,
        const int treeIndex,
        const std::string& sourceName,
        const InlineModelTransform& transform,
        const std::vector<int>* onlyPartitionIndices,
        const std::vector<int>* excludedPartitionIndices)
    {
        if (treeIndex < 0 || treeIndex >= clipMap.aabbTreeCount)
            return;

        const auto& tree = clipMap.aabbTrees[treeIndex];
        if (tree.childCount != 0u)
        {
            // Mirror CM_TraceThroughAabbTree_r: internal AABB nodes do not own
            // triangles directly; their contiguous children eventually lead to
            // leaf nodes that hold CollisionPartition indices.
            for (auto childOffset = 0u; childOffset < tree.childCount; childOffset++)
                AppendCollisionTreeTriangles(
                    byMaterial,
                    emittedTriangles,
                    clipMap,
                    tree.u.firstChildIndex + static_cast<int>(childOffset),
                    sourceName,
                    transform,
                    onlyPartitionIndices,
                    excludedPartitionIndices);

            return;
        }

        const auto partitionIndex = tree.u.partitionIndex;
        if (partitionIndex < 0 || partitionIndex >= clipMap.partitionCount)
            return;

        // onlyPartitionIndices is used for brushmodel non-brush export. The
        // source comes from cmodel.leaf, but ownership is filtered by partition
        // because CM_TraceThroughAabbTree_work stamps partitionIndex, not treeIndex.
        if (onlyPartitionIndices != nullptr && !IsExcludedIndex(partitionIndex, *onlyPartitionIndices))
            return;

        // excludedPartitionIndices is used for world export so model-owned
        // non-brush partitions are not emitted again under world/nonbrush.
        if (excludedPartitionIndices != nullptr && IsExcludedIndex(partitionIndex, *excludedPartitionIndices))
            return;

        const auto& partition = clipMap.partitions[partitionIndex];
        // CMod_LoadCollisionPartitions @ .text:0069D730 asserts this exact range:
        // out->firstTri + out->triCount <= cm.triCount and out->firstTri >= 0.
        if (partition.firstTri < 0 || partition.firstTri + static_cast<int>(partition.triCount) > clipMap.triCount)
            throw std::runtime_error(
                "Collision partition has invalid triangle range partitionIndex="
                + std::to_string(partitionIndex)
                + " firstTri="
                + std::to_string(partition.firstTri)
                + " triCount="
                + std::to_string(static_cast<unsigned>(partition.triCount))
                + " clipMapTriCount="
                + std::to_string(clipMap.triCount));

        // CM_TraceThroughAabbTree @ .text:006A1C60 reads
        // cm.info.materials[aabbTree->materialIndex], checks contentFlags
        // against tw->contentsMask, and writes surfaceFlags/contentFlags to trace.
        if (clipMap.info.materials == nullptr || tree.materialIndex >= clipMap.info.numMaterials)
            throw std::runtime_error(
                "Collision AABB tree has invalid material index treeIndex="
                + std::to_string(treeIndex)
                + " materialIndex="
                + std::to_string(tree.materialIndex)
                + " numMaterials="
                + std::to_string(clipMap.info.numMaterials));

        auto& mesh = byMaterial[std::to_string(tree.materialIndex)];
        if (mesh.name.empty())
        {
            const auto& material = clipMap.info.materials[tree.materialIndex];
            const auto materialName = material.name && material.name[0] ? material.name : "unknown";

            mesh.name = materialName;
            mesh.groupName = sourceName + "/nonbrush";
            mesh.color = ColorForIndex(tree.materialIndex);
        }

        // CM_TraceThroughAabbTree_work @ .text:006A1430 computes:
        // triIndices + (partition.firstTri * 3 * sizeof(uint16_t)), then advances
        // six bytes per triangle. The T6 runtime struct is uint16_t[3].
        for (auto triOffset = 0u; triOffset < partition.triCount; triOffset++)
        {
            const auto triIndex = partition.firstTri + static_cast<int>(triOffset);
            if (emittedTriangles[static_cast<size_t>(triIndex)])
                continue;

            const auto& tri = clipMap.triIndices[triIndex];
            if (tri[0] >= clipMap.vertCount || tri[1] >= clipMap.vertCount || tri[2] >= clipMap.vertCount)
            {
                throw std::runtime_error(
                    "Collision triangle has invalid vertex index triIndex="
                    + std::to_string(triIndex)
                    + " indices="
                    + std::to_string(tri[0])
                    + ","
                    + std::to_string(tri[1])
                    + ","
                    + std::to_string(tri[2])
                    + " vertCount="
                    + std::to_string(clipMap.vertCount));
            }

            const auto a = TransformPoint(clipMap.verts[tri[0]], transform);
            const auto b = TransformPoint(clipMap.verts[tri[1]], transform);
            const auto c = TransformPoint(clipMap.verts[tri[2]], transform);
            AppendTriangle(mesh, a, b, c);
            emittedTriangles[static_cast<size_t>(triIndex)] = true;
        }
    }

    std::vector<MeshPrimitive> BuildNonBrushCollisionMeshes(
        const clipMap_t& clipMap,
        const std::string& sourceName,
        const InlineModelTransform& transform,
        const std::vector<int>& onlyTreeIndices = {},
        const std::vector<int>& excludedTreeIndices = {},
        const bool includeUnclassifiedTriangles = true,
        const std::vector<int>* onlyPartitionIndices = nullptr,
        const std::vector<int>* excludedPartitionIndices = nullptr)
    {
        if (clipMap.verts == nullptr || clipMap.triIndices == nullptr || clipMap.vertCount == 0u || clipMap.triCount <= 0)
            return {};

        std::map<std::string, MeshPrimitive> byMaterial;
        std::vector<bool> emittedTriangles;
        emittedTriangles.resize(static_cast<size_t>(clipMap.triCount));

        // CM_TraceThroughLeaf @ .text:006A62C0 walks a cLeaf_s AABB range:
        // firstCollAabbIndex at +0x00, collAabbCount at +0x02. CM_Trace passes
        // cmodel + 0x20 as that leaf for model handles, while world traces reach
        // leaves through CM_TraceThroughTree @ .text:006A7230.
        if (clipMap.aabbTrees && clipMap.partitions)
        {
            if (!onlyTreeIndices.empty())
            {
                for (const auto treeIndex : onlyTreeIndices)
                    AppendCollisionTreeTriangles(
                        byMaterial,
                        emittedTriangles,
                        clipMap,
                        treeIndex,
                        sourceName,
                        transform,
                        onlyPartitionIndices,
                        excludedPartitionIndices);
            }
            else
            {
                for (auto treeIndex = 0; treeIndex < clipMap.aabbTreeCount; treeIndex++)
                {
                    if (IsExcludedIndex(treeIndex, excludedTreeIndices))
                        continue;

                    AppendCollisionTreeTriangles(
                        byMaterial,
                        emittedTriangles,
                        clipMap,
                        treeIndex,
                        sourceName,
                        transform,
                        onlyPartitionIndices,
                        excludedPartitionIndices);
                }
            }
        }

        if (includeUnclassifiedTriangles)
        {
            for (auto triIndex = 0; triIndex < clipMap.triCount; triIndex++)
            {
                if (emittedTriangles[static_cast<size_t>(triIndex)])
                    continue;

                const auto& tri = clipMap.triIndices[triIndex];
                if (tri[0] >= clipMap.vertCount || tri[1] >= clipMap.vertCount || tri[2] >= clipMap.vertCount)
                    continue;

                auto& mesh = byMaterial["unclassified"];
                if (mesh.name.empty())
                {
                    mesh.name = "unclassified";
                    mesh.groupName = sourceName + "/nonbrush";
                    mesh.color = {0.7f, 0.7f, 0.7f};
                }
                const auto a = TransformPoint(clipMap.verts[tri[0]], transform);
                const auto b = TransformPoint(clipMap.verts[tri[1]], transform);
                const auto c = TransformPoint(clipMap.verts[tri[2]], transform);
                AppendTriangle(mesh, a, b, c);
            }
        }

        std::vector<MeshPrimitive> result;
        for (auto& entry : byMaterial)
        {
            if (!entry.second.indices.empty())
                result.emplace_back(std::move(entry.second));
        }

        return result;
    }

    MeshPrimitive BuildBrushFaceMesh(const cbrush_t& brush, const cbrushside_t& brushSide, const std::string& name, const vec3_t color)
    {
        MeshPrimitive result{name, {}, {}, {}, {}, color};

        if (brush.verts == nullptr || brush.numverts < 3u || brushSide.plane == nullptr)
            return result;

        result.vertices.reserve(brush.numverts);
        result.uvs.reserve(brush.numverts);
        for (auto vertexIndex = 0u; vertexIndex < brush.numverts; vertexIndex++)
        {
            result.vertices.emplace_back(brush.verts[vertexIndex]);
            result.uvs.emplace_back(PlanarUvForVertex(result.vertices.back(), brushSide.plane->normal));
        }

        std::vector<uint32_t> faceIndices;

        for (auto vertexIndex = 0u; vertexIndex < result.vertices.size(); vertexIndex++)
        {
            if (NearlyEqual(Dot(result.vertices[vertexIndex], brushSide.plane->normal), brushSide.plane->dist))
                faceIndices.emplace_back(static_cast<uint32_t>(vertexIndex));
        }

        if (faceIndices.size() < 3u)
            return result;

        SortFaceVertices(faceIndices, result.vertices, brushSide.plane->normal);

        for (auto faceVertexIndex = 1u; faceVertexIndex + 1u < faceIndices.size(); faceVertexIndex++)
        {
            result.indices.emplace_back(faceIndices[0]);
            result.indices.emplace_back(faceIndices[faceVertexIndex]);
            result.indices.emplace_back(faceIndices[faceVertexIndex + 1u]);
        }

        return result;
    }

    void AppendMesh(MeshPrimitive& target, const MeshPrimitive& source)
    {
        const auto baseIndex = static_cast<uint32_t>(target.vertices.size());
        target.vertices.insert(target.vertices.end(), source.vertices.begin(), source.vertices.end());
        target.uvs.insert(target.uvs.end(), source.uvs.begin(), source.uvs.end());

        for (const auto index : source.indices)
            target.indices.emplace_back(baseIndex + index);
    }

    void TransformMesh(MeshPrimitive& mesh, const InlineModelTransform& transform)
    {
        const auto hasRotation = !IsZero(transform.angles);
        std::array<vec3_t, 3> axis{};
        if (hasRotation)
            AnglesToAxis(transform.angles, axis);

        for (auto& vertex : mesh.vertices)
        {
            if (hasRotation)
                vertex = RotateByAxis(vertex, axis);

            vertex = vertex + transform.origin;
        }
    }

    void SkipWhitespace(const char*& current, const char* end)
    {
        while (current < end && std::isspace(static_cast<unsigned char>(*current)))
            current++;
    }

    std::string ParseQuotedString(const char*& current, const char* end)
    {
        SkipWhitespace(current, end);
        if (current >= end || *current != '"')
            return {};

        current++;
        std::string value;
        while (current < end && *current != '"')
        {
            value.push_back(*current);
            current++;
        }

        if (current < end && *current == '"')
            current++;

        return value;
    }

    std::vector<ParsedMapEntity> ParseMapEntities(const MapEnts* mapEnts)
    {
        std::vector<ParsedMapEntity> entities;
        if (mapEnts == nullptr || mapEnts->entityString == nullptr || mapEnts->numEntityChars <= 0)
            return entities;

        const auto* current = mapEnts->entityString;
        const auto* end = current + mapEnts->numEntityChars;
        while (current < end)
        {
            SkipWhitespace(current, end);
            if (current >= end)
                break;

            if (*current != '{')
            {
                current++;
                continue;
            }

            current++;
            ParsedMapEntity entity;
            while (current < end)
            {
                SkipWhitespace(current, end);
                if (current >= end)
                    break;

                if (*current == '}')
                {
                    current++;
                    break;
                }

                auto key = ParseQuotedString(current, end);
                auto value = ParseQuotedString(current, end);
                if (!key.empty())
                    entity.values.emplace(std::move(key), std::move(value));
            }

            entities.emplace_back(std::move(entity));
        }

        return entities;
    }

    bool ParseVec3(const std::string& value, vec3_t& result)
    {
        std::istringstream stream(value);
        return static_cast<bool>(stream >> result.x >> result.y >> result.z);
    }

    std::map<unsigned, InlineModelTransform> BuildInlineModelTransforms(const MapEnts* mapEnts)
    {
        std::map<unsigned, InlineModelTransform> transforms;
        for (const auto& entity : ParseMapEntities(mapEnts))
        {
            const auto model = entity.values.find("model");
            const auto origin = entity.values.find("origin");
            if (model == entity.values.end() || model->second.size() < 2u || model->second[0] != '*')
                continue;

            InlineModelTransform transform{};
            if (origin != entity.values.end())
                ParseVec3(origin->second, transform.origin);

            const auto angles = entity.values.find("angles");
            if (angles != entity.values.end())
                ParseVec3(angles->second, transform.angles);
            transform.values = entity.values;

            try
            {
                const auto modelIndex = static_cast<unsigned>(std::stoul(model->second.substr(1)));
                transforms.emplace(modelIndex, transform);
            }
            catch (const std::exception&)
            {
            }
        }

        return transforms;
    }

    bool TryGetModelIndexFromGroupName(const std::string& groupName, unsigned& modelIndex)
    {
        constexpr std::string_view PREFIX = "model_*";
        if (groupName.compare(0u, PREFIX.size(), PREFIX) != 0)
            return false;

        auto indexEnd = PREFIX.size();
        while (indexEnd < groupName.size() && std::isdigit(static_cast<unsigned char>(groupName[indexEnd])))
            indexEnd++;

        if (indexEnd == PREFIX.size())
            return false;

        try
        {
            modelIndex = static_cast<unsigned>(std::stoul(groupName.substr(PREFIX.size(), indexEnd - PREFIX.size())));
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    nlohmann::ordered_json MapValuesToExtras(const unsigned modelIndex, const InlineModelTransform& transform)
    {
        nlohmann::ordered_json extras;
        extras["modelIndex"] = modelIndex;

        for (const auto& value : transform.values)
            extras["mapEnt_" + value.first] = value.second;

        return extras;
    }

    // Expects each material to have a unique content/surface flag combination. It seems to be correct.
    const ClipMaterial& GetClipMaterialByFlags(const ClipInfo& clipInfo, const int contentFlags, const int surfaceFlags)
    {
        if (clipInfo.numMaterials == 0u || clipInfo.materials == nullptr)
        {
            throw std::runtime_error("Clipinfo has no materials");
        }

        for (auto materialIndex = 0u; materialIndex < clipInfo.numMaterials; materialIndex++)
        {
            const auto& material = clipInfo.materials[materialIndex];

            if (material.contentFlags == contentFlags && material.surfaceFlags == surfaceFlags)
            {
                return material;
            }
        }

        throw std::runtime_error("Could not find clip material for contentFlags=0x" + Hex(static_cast<uint32_t>(contentFlags))
                                 + " surfaceFlags=0x" + Hex(static_cast<uint32_t>(surfaceFlags)));
    }

    std::string GetBrushSideMaterialName(const ClipInfo& clipInfo, const cbrushside_t& brushSide)
    {
        const auto& material = GetClipMaterialByFlags(clipInfo, brushSide.cflags, brushSide.sflags);
        if (material.name == nullptr || material.name[0] == '\0')
            throw std::runtime_error("Clip material has no name for contentFlags=0x" + Hex(static_cast<uint32_t>(brushSide.cflags))
                                     + " surfaceFlags=0x" + Hex(static_cast<uint32_t>(brushSide.sflags)));

        return material.name;
    }

    bool SameVec3(const vec3_t& a, const vec3_t& b)
    {
        return NearlyEqual(a.x, b.x) && NearlyEqual(a.y, b.y) && NearlyEqual(a.z, b.z);
    }

    bool SamePlane(const cplane_s* a, const cplane_s* b)
    {
        if (a == nullptr || b == nullptr)
            return a == b;

        return SameVec3(a->normal, b->normal) && NearlyEqual(a->dist, b->dist);
    }

    bool SameBrushShape(const cbrush_t& a, const cbrush_t& b)
    {
        if (!SameVec3(a.mins, b.mins) || !SameVec3(a.maxs, b.maxs) || a.contents != b.contents || a.numsides != b.numsides)
            return false;

        for (auto sideIndex = 0u; sideIndex < a.numsides; sideIndex++)
        {
            const auto* aSide = a.sides != nullptr ? &a.sides[sideIndex] : nullptr;
            const auto* bSide = b.sides != nullptr ? &b.sides[sideIndex] : nullptr;
            if (aSide == nullptr || bSide == nullptr)
            {
                if (aSide != bSide)
                    return false;

                continue;
            }

            if (aSide->cflags != bSide->cflags || aSide->sflags != bSide->sflags || !SamePlane(aSide->plane, bSide->plane))
                return false;
        }

        return true;
    }

    void CollectBrushIndicesForLeafBrushNode(const ClipInfo& clipInfo, const int nodeIndex, std::vector<unsigned>& brushIndices, std::vector<bool>& visitedNodes)
    {
        if (clipInfo.leafbrushNodes == nullptr || nodeIndex < 0 || static_cast<unsigned>(nodeIndex) >= clipInfo.leafbrushNodesCount)
            return;

        if (visitedNodes[static_cast<size_t>(nodeIndex)])
            return;

        visitedNodes[static_cast<size_t>(nodeIndex)] = true;

        const auto& node = clipInfo.leafbrushNodes[nodeIndex];
        if (node.leafBrushCount > 0)
        {
            if (node.data.leaf.brushes == nullptr)
                return;

            for (auto brushOffset = 0; brushOffset < node.leafBrushCount; brushOffset++)
            {
                const auto brushIndex = static_cast<unsigned>(node.data.leaf.brushes[brushOffset]);
                if (brushIndex < clipInfo.numBrushes)
                    brushIndices.emplace_back(brushIndex);
            }

            return;
        }

        const auto childZeroIndex = nodeIndex + node.data.children.childOffset[0];
        const auto childOneIndex = nodeIndex + node.data.children.childOffset[1];
        CollectBrushIndicesForLeafBrushNode(clipInfo, childZeroIndex, brushIndices, visitedNodes);
        CollectBrushIndicesForLeafBrushNode(clipInfo, childOneIndex, brushIndices, visitedNodes);
    }

    const std::vector<unsigned>& GetBrushIndicesForLeafBrushNodeCached(LeafBrushNodeCache& cache, const ClipInfo& clipInfo, const int leafBrushNode)
    {
        const auto existingEntry = cache.find(leafBrushNode);
        if (existingEntry != cache.end())
            return existingEntry->second;

        std::vector<unsigned> brushIndices;
        std::vector<bool> visitedNodes(clipInfo.leafbrushNodesCount);
        CollectBrushIndicesForLeafBrushNode(clipInfo, leafBrushNode, brushIndices, visitedNodes);
        std::sort(brushIndices.begin(), brushIndices.end());
        brushIndices.erase(std::unique(brushIndices.begin(), brushIndices.end()), brushIndices.end());

        const auto insertedEntry = cache.emplace(leafBrushNode, std::move(brushIndices));
        return insertedEntry.first->second;
    }

    bool IsExcludedBrushIndex(const unsigned brushIndex, const std::vector<unsigned>& excludedBrushIndices)
    {
        return std::binary_search(excludedBrushIndices.begin(), excludedBrushIndices.end(), brushIndex);
    }

    std::vector<unsigned> GetAllBrushIndices(const ClipInfo& clipInfo)
    {
        std::vector<unsigned> brushIndices;
        if (clipInfo.brushes == nullptr)
            return brushIndices;

        brushIndices.reserve(clipInfo.numBrushes);
        for (auto brushIndex = 0u; brushIndex < clipInfo.numBrushes; brushIndex++)
            brushIndices.emplace_back(brushIndex);

        return brushIndices;
    }

    std::vector<int> GetAabbTreeIndicesForLeaf(const clipMap_t& clipMap, const cLeaf_s& leaf)
    {
        std::vector<int> treeIndices;
        if (clipMap.aabbTrees == nullptr || leaf.collAabbCount == 0u)
            return treeIndices;

        // CM_TraceThroughLeaf @ .text:006A62C0 iterates:
        // for i in [0, leaf.collAabbCount): cm.aabbTrees[leaf.firstCollAabbIndex + i].
        treeIndices.reserve(leaf.collAabbCount);
        for (auto treeOffset = 0u; treeOffset < leaf.collAabbCount; treeOffset++)
        {
            const auto treeIndex = static_cast<int>(leaf.firstCollAabbIndex + treeOffset);
            if (treeIndex >= 0 && treeIndex < clipMap.aabbTreeCount)
                treeIndices.emplace_back(treeIndex);
        }

        return treeIndices;
    }

    std::vector<int> GetWorldAabbTreeRootIndices(const clipMap_t& clipMap)
    {
        std::vector<int> treeIndices;
        if (clipMap.leafs == nullptr)
            return treeIndices;

        for (auto leafIndex = 0u; leafIndex < clipMap.numLeafs; leafIndex++)
        {
            auto leafTreeIndices = GetAabbTreeIndicesForLeaf(clipMap, clipMap.leafs[leafIndex]);
            treeIndices.insert(treeIndices.end(), leafTreeIndices.begin(), leafTreeIndices.end());
        }

        std::sort(treeIndices.begin(), treeIndices.end());
        treeIndices.erase(std::unique(treeIndices.begin(), treeIndices.end()), treeIndices.end());
        return treeIndices;
    }

    void ExpandBrushIndicesToContiguousRange(std::vector<unsigned>& brushIndices)
    {
        if (brushIndices.empty())
            return;

        const auto minMax = std::minmax_element(brushIndices.begin(), brushIndices.end());
        std::vector<unsigned> expandedBrushIndices;
        expandedBrushIndices.reserve(*minMax.second - *minMax.first + 1u);
        for (auto brushIndex = *minMax.first; brushIndex <= *minMax.second; brushIndex++)
            expandedBrushIndices.emplace_back(brushIndex);

        brushIndices = std::move(expandedBrushIndices);
    }

    std::vector<unsigned> RecoverSubmodelBrushRange(LeafBrushNodeCache& cache, const ClipInfo& clipInfo, const cmodel_t& cmodel)
    {
        auto brushIndices = GetBrushIndicesForLeafBrushNodeCached(cache, clipInfo, cmodel.leaf.leafBrushNode);

        // CMod_LoadSubmodelBrushNodes @ .text:0069CB80 reads firstBrush and
        // numBrushes from the BSP submodel lump, fills a temporary list with
        // firstBrush + i, then calls CMod_PartionLeafBrushes @ .text:0069C9A0.
        // The generated leafBrushNode is an acceleration tree, so recovering
        // the original submodel brush ownership means expanding back to that
        // contiguous firstBrush..firstBrush+numBrushes range.
        ExpandBrushIndicesToContiguousRange(brushIndices);
        return brushIndices;
    }

    bool UsesWorldBrushArray(const clipMap_t& clipMap, const ClipInfo* clipInfo)
    {
        if (clipInfo == nullptr)
            return false;

        return clipInfo == &clipMap.info || clipInfo == clipMap.pInfo || clipInfo->brushes == clipMap.info.brushes;
    }

    BrushModelBrushes ResolveBrushModelBrushes(LeafBrushNodeCache& cache, const clipMap_t& clipMap, const cmodel_t& cmodel)
    {
        BrushModelBrushes result{
            &clipMap.info,
            RecoverSubmodelBrushRange(cache, clipMap.info, cmodel),
            "recovered_submodel_range",
            false,
        };

        if (result.indices.empty() && cmodel.info != nullptr && !UsesWorldBrushArray(clipMap, cmodel.info))
        {
            result.clipInfo = cmodel.info;
            result.indices = GetAllBrushIndices(*cmodel.info);
            result.source = "cmodel_clipinfo";
            result.usesOwnClipInfo = !result.indices.empty();
        }

        return result;
    }

    bool IsExcludedBrush(const cbrush_t& brush, const std::vector<const cbrush_t*>& excludedBrushes)
    {
        for (const auto* excludedBrush : excludedBrushes)
        {
            if (excludedBrush == nullptr)
                continue;

            if (&brush == excludedBrush || SameBrushShape(brush, *excludedBrush))
                return true;
        }

        return false;
    }

    void AppendClipInfoBrushes(
        std::vector<MeshPrimitive>& primitives,
        const ClipInfo& clipInfo,
        const std::string& sourceName,
        const InlineModelTransform& transform,
        const std::vector<const cbrush_t*>& excludedBrushes = {},
        const std::vector<unsigned>& excludedBrushIndices = {},
        const std::vector<unsigned>* onlyBrushIndices = nullptr,
        const bool separateBrushes = false,
        const bool useLocalBrushNames = false,
        const bool groupByContents = true)
    {
        if (clipInfo.brushes == nullptr)
            return;

        std::map<std::string, MeshPrimitive> brushesByMaterialAndContents;
        auto localBrushIndex = 0u;
        const auto exportBrush = [&](const unsigned brushIndex)
        {
            if (brushIndex >= clipInfo.numBrushes)
                return;

            if (IsExcludedBrushIndex(brushIndex, excludedBrushIndices))
                return;

            const auto& brush = clipInfo.brushes[brushIndex];
            if (IsExcludedBrush(brush, excludedBrushes))
                return;

            auto brushGroupName = sourceName;
            if (groupByContents && !MERGE_COLLISION_TYPES)
            {
                const auto contents = clipInfo.brushContents ? clipInfo.brushContents[brushIndex] : brush.contents;
                brushGroupName += "/contents_0x" + Hex(static_cast<uint32_t>(contents));
            }

            if (separateBrushes)
                brushGroupName += "/brush_" + std::to_string(useLocalBrushNames ? localBrushIndex : brushIndex);

            localBrushIndex++;

            std::array<cplane_s, 6> axialPlanes{};
            const auto brushSides = BuildBrushSides(brush, axialPlanes);

            for (const auto& brushSide : brushSides)
            {
                const auto materialName = GetBrushSideMaterialName(clipInfo, brushSide);
                const auto mapKey = brushGroupName + "_" + materialName;
                auto& groupedBrushMesh = brushesByMaterialAndContents[mapKey];
                if (groupedBrushMesh.name.empty())
                {
                    groupedBrushMesh.name = materialName;
                    groupedBrushMesh.groupName = brushGroupName;
                    groupedBrushMesh.color = ColorForIndex(static_cast<unsigned>(brushesByMaterialAndContents.size() + 8u));
                }

                auto brushMesh = BuildBrushFaceMesh(brush, brushSide, groupedBrushMesh.name, groupedBrushMesh.color);
                if (!brushMesh.indices.empty())
                {
                    TransformMesh(brushMesh, transform);
                    AppendMesh(groupedBrushMesh, brushMesh);
                }
            }
        };

        if (onlyBrushIndices != nullptr)
        {
            for (const auto brushIndex : *onlyBrushIndices)
                exportBrush(brushIndex);
        }
        else
        {
            for (auto brushIndex = 0u; brushIndex < clipInfo.numBrushes; brushIndex++)
                exportBrush(brushIndex);
        }

        for (auto& entry : brushesByMaterialAndContents)
        {
            if (!entry.second.indices.empty())
                primitives.emplace_back(std::move(entry.second));
        }
    }

    void WriteUint32(std::ostream& stream, const uint32_t value)
    {
        stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    void WriteGlb(
        std::ostream& glbStream,
        const std::vector<MeshPrimitive>& primitives,
        const std::map<unsigned, InlineModelTransform>& inlineModelTransforms,
        const std::map<std::string, MaterialTextureNames>& materialTextureLookup)
    {
        using namespace gltf;

        std::vector<uint8_t> buffer;

        struct PrimitiveAccessors
        {
            const MeshPrimitive* primitive;
            unsigned position;
            unsigned texCoord;
            unsigned indices;
            unsigned material;
        };
        std::vector<PrimitiveAccessors> primitiveAccessors;
        std::map<std::string, unsigned> materialIndices;
        const auto addNodeMaterialExtras = [&](nlohmann::ordered_json& extras, const std::vector<unsigned>& primitiveIndices)
        {
            nlohmann::ordered_json materials = nlohmann::ordered_json::array();
            std::vector<std::string> seenMaterials;
            for (const auto primitiveIndex : primitiveIndices)
            {
                const auto& materialName = primitiveAccessors[primitiveIndex].primitive->name;
                if (std::find(seenMaterials.begin(), seenMaterials.end(), materialName) != seenMaterials.end())
                    continue;

                seenMaterials.emplace_back(materialName);
                materials.emplace_back(materialName);
            }

            extras["materialCount"] = static_cast<unsigned>(seenMaterials.size());
            extras["materials"] = std::move(materials);
        };

        JsonRoot gltf;
        gltf.asset.version = "2.0";
        gltf.asset.generator = "OpenAssetTools";
        gltf.bufferViews.emplace();
        gltf.accessors.emplace();
        gltf.materials.emplace();
        gltf.meshes.emplace();
        gltf.nodes.emplace();

        const auto createTexture = [&](const std::string& textureName, const std::string_view extension = "dds")
        {
            if (!gltf.textures.has_value())
                gltf.textures.emplace();
            if (!gltf.images.has_value())
                gltf.images.emplace();

            const auto uri = "../../images/" + textureName + "." + std::string(extension);

            auto existingTextureIndex = 0u;
            for (const auto& existingTexture : gltf.textures.value())
            {
                const auto& existingImage = gltf.images.value()[existingTexture.source];
                if (existingImage.uri == uri)
                    return existingTextureIndex;

                existingTextureIndex++;
            }

            JsonImage image;
            image.uri = uri;
            const auto imageIndex = static_cast<unsigned>(gltf.images->size());
            gltf.images->emplace_back(std::move(image));

            JsonTexture texture;
            texture.source = imageIndex;
            const auto textureIndex = static_cast<unsigned>(gltf.textures->size());
            gltf.textures->emplace_back(texture);

            return textureIndex;
        };

        const auto addBufferView = [&](const void* data, const size_t dataSize, const JsonBufferViewTarget target)
        {
            PadBuffer(buffer, 4u);
            const auto offset = buffer.size();
            AppendBytes(buffer, data, dataSize);

            JsonBufferView bufferView;
            bufferView.buffer = 0u;
            bufferView.byteOffset = static_cast<unsigned>(offset);
            bufferView.byteLength = static_cast<unsigned>(dataSize);
            bufferView.target = target;

            const auto bufferViewIndex = static_cast<unsigned>(gltf.bufferViews->size());
            gltf.bufferViews->emplace_back(std::move(bufferView));
            return bufferViewIndex;
        };

        for (const auto& primitive : primitives)
        {
            if (primitive.vertices.empty() || primitive.indices.empty())
                continue;

            const auto materialKey = primitive.name;
            auto materialIndex = 0u;
            const auto existingMaterial = materialIndices.find(materialKey);
            if (existingMaterial != materialIndices.end())
            {
                materialIndex = existingMaterial->second;
            }
            else
            {
                materialIndex = static_cast<unsigned>(gltf.materials->size());
                materialIndices.emplace(materialKey, materialIndex);

                JsonPbrMetallicRoughness roughness;
                roughness.metallicFactor = 0.0f;
                roughness.roughnessFactor = 1.0f;

                JsonMaterial material;
                material.name = primitive.name;
                material.doubleSided = true;
                const auto materialTextures = materialTextureLookup.find(primitive.name);
                if (materialTextures != materialTextureLookup.end())
                {
                    if (!materialTextures->second.colorMapName.empty())
                    {
                        roughness.baseColorTexture.emplace();
                        roughness.baseColorTexture->index = createTexture(materialTextures->second.colorMapName);
                    }
                }
                else if (!primitive.name.empty())
                {
                    con::warn("Clipmap material \"{}\" is missing; falling back to ../../images/{}.tif", primitive.name, primitive.name);
                    roughness.baseColorTexture.emplace();
                    roughness.baseColorTexture->index = createTexture(primitive.name, "tif");
                    roughness.baseColorFactor = std::vector{1.0f, 1.0f, 1.0f, 0.1f};
                    material.alphaMode = "BLEND";
                }

                material.pbrMetallicRoughness = std::move(roughness);
                gltf.materials->emplace_back(std::move(material));
            }

            vec3_t min = primitive.vertices[0];
            vec3_t max = primitive.vertices[0];
            for (const auto& vertex : primitive.vertices)
            {
                min.x = std::min(min.x, vertex.x);
                min.y = std::min(min.y, vertex.y);
                min.z = std::min(min.z, vertex.z);
                max.x = std::max(max.x, vertex.x);
                max.y = std::max(max.y, vertex.y);
                max.z = std::max(max.z, vertex.z);
            }

            const auto positionBufferView = addBufferView(
                primitive.vertices.data(), primitive.vertices.size() * sizeof(vec3_t), JsonBufferViewTarget::ARRAY_BUFFER);

            JsonAccessor positionAccessor;
            positionAccessor.bufferView = positionBufferView;
            positionAccessor.componentType = JsonAccessorComponentType::FLOAT;
            positionAccessor.count = static_cast<unsigned>(primitive.vertices.size());
            positionAccessor.type = JsonAccessorType::VEC3;
            positionAccessor.min = std::vector{min.x, min.y, min.z};
            positionAccessor.max = std::vector{max.x, max.y, max.z};
            const auto positionAccessorIndex = static_cast<unsigned>(gltf.accessors->size());
            gltf.accessors->emplace_back(std::move(positionAccessor));

            const auto texCoordBufferView = addBufferView(primitive.uvs.data(), primitive.uvs.size() * sizeof(vec2_t), JsonBufferViewTarget::ARRAY_BUFFER);
            JsonAccessor texCoordAccessor;
            texCoordAccessor.bufferView = texCoordBufferView;
            texCoordAccessor.componentType = JsonAccessorComponentType::FLOAT;
            texCoordAccessor.count = static_cast<unsigned>(primitive.uvs.size());
            texCoordAccessor.type = JsonAccessorType::VEC2;
            const auto texCoordAccessorIndex = static_cast<unsigned>(gltf.accessors->size());
            gltf.accessors->emplace_back(std::move(texCoordAccessor));

            const auto indexBufferView = addBufferView(primitive.indices.data(), primitive.indices.size() * sizeof(uint32_t), JsonBufferViewTarget::ELEMENT_ARRAY_BUFFER);
            JsonAccessor indexAccessor;
            indexAccessor.bufferView = indexBufferView;
            indexAccessor.componentType = JsonAccessorComponentType::UNSIGNED_INT;
            indexAccessor.count = static_cast<unsigned>(primitive.indices.size());
            indexAccessor.type = JsonAccessorType::SCALAR;
            const auto indexAccessorIndex = static_cast<unsigned>(gltf.accessors->size());
            gltf.accessors->emplace_back(std::move(indexAccessor));

            primitiveAccessors.push_back({&primitive, positionAccessorIndex, texCoordAccessorIndex, indexAccessorIndex, materialIndex});
        }

        std::map<std::string, std::vector<unsigned>> primitivesByGroup;
        for (auto i = 0u; i < primitiveAccessors.size(); i++)
        {
            const auto& primitive = *primitiveAccessors[i].primitive;
            primitivesByGroup[primitive.groupName.empty() ? "ungrouped" : primitive.groupName].emplace_back(i);
        }

        JsonNode rootNode;
        rootNode.name = "clipmap";
        rootNode.children.emplace();

        const auto firstMeshNodeIndex = 1u;
        for (auto groupIndex = 0u; groupIndex < primitivesByGroup.size(); groupIndex++)
            rootNode.children->emplace_back(firstMeshNodeIndex + groupIndex);

        gltf.nodes->emplace_back(std::move(rootNode));

        for (const auto& entry : primitivesByGroup)
        {
            JsonMesh mesh;
            for (const auto primitiveIndex : entry.second)
            {
                const auto& primitive = primitiveAccessors[primitiveIndex];
                JsonMeshPrimitives meshPrimitive;
                meshPrimitive.attributes.POSITION = primitive.position;
                meshPrimitive.attributes.TEXCOORD_0 = primitive.texCoord;
                meshPrimitive.indices = primitive.indices;
                meshPrimitive.material = primitive.material;
                meshPrimitive.mode = JsonMeshPrimitivesMode::TRIANGLES;
                mesh.primitives.emplace_back(std::move(meshPrimitive));
            }

            const auto meshIndex = static_cast<unsigned>(gltf.meshes->size());
            gltf.meshes->emplace_back(std::move(mesh));

            JsonNode meshNode;
            meshNode.name = entry.first;
            meshNode.mesh = meshIndex;
            nlohmann::ordered_json extras;
            addNodeMaterialExtras(extras, entry.second);

            unsigned modelIndex = 0u;
            if (TryGetModelIndexFromGroupName(entry.first, modelIndex))
            {
                const auto transform = inlineModelTransforms.find(modelIndex);
                if (transform != inlineModelTransforms.end())
                {
                    auto mapEntExtras = MapValuesToExtras(modelIndex, transform->second);
                    for (auto& value : mapEntExtras.items())
                        extras[value.key()] = value.value();
                }
            }
            meshNode.extras = std::move(extras);
            gltf.nodes->emplace_back(std::move(meshNode));
        }

        JsonBuffer gltfBuffer;
        gltfBuffer.byteLength = static_cast<unsigned>(buffer.size());
        gltf.buffers = std::vector{std::move(gltfBuffer)};

        JsonScene scene;
        scene.nodes.emplace_back(0u);
        gltf.scenes = std::vector{std::move(scene)};
        gltf.scene = 0u;

        nlohmann::ordered_json json = gltf;
        auto jsonText = json.dump();
        while (jsonText.size() % 4u != 0u)
            jsonText.push_back(' ');

        while (buffer.size() % 4u != 0u)
            buffer.emplace_back(0u);

        constexpr uint32_t GLB_MAGIC = 0x46546C67u;
        constexpr uint32_t GLB_VERSION = 2u;
        constexpr uint32_t JSON_CHUNK_TYPE = 0x4E4F534Au;
        constexpr uint32_t BIN_CHUNK_TYPE = 0x004E4942u;

        const auto jsonChunkLength = static_cast<uint32_t>(jsonText.size());
        const auto binChunkLength = static_cast<uint32_t>(buffer.size());
        const auto totalLength = static_cast<uint32_t>(12u + 8u + jsonChunkLength + (binChunkLength > 0u ? 8u + binChunkLength : 0u));

        WriteUint32(glbStream, GLB_MAGIC);
        WriteUint32(glbStream, GLB_VERSION);
        WriteUint32(glbStream, totalLength);
        WriteUint32(glbStream, jsonChunkLength);
        WriteUint32(glbStream, JSON_CHUNK_TYPE);
        glbStream.write(jsonText.data(), static_cast<std::streamsize>(jsonText.size()));

        if (binChunkLength > 0u)
        {
            WriteUint32(glbStream, binChunkLength);
            WriteUint32(glbStream, BIN_CHUNK_TYPE);
            glbStream.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        }
    }

} // namespace

namespace clip_map
{
    void DumperT6::DumpAsset(AssetDumpingContext& context, const XAssetInfo<AssetClipMapPvs::Type>& asset)
    {
        const auto* clipMap = asset.Asset();
        const auto glbName = asset.m_name + ".clipmap.glb";
        const auto glbFile = context.OpenAssetFile(glbName);

        if (!glbFile)
            return;

        std::vector<MeshPrimitive> primitives;

        const auto inlineModelTransforms = BuildInlineModelTransforms(clipMap->mapEnts);
        std::vector<const cbrush_t*> inlineModelBrushes;
        std::vector<unsigned> inlineModelWorldBrushIndices;
        std::vector<int> worldBrushModelOwner(clipMap->info.numBrushes, -1);
        std::vector<int> inlineModelAabbTreeIndices;
        std::vector<int> inlineModelPartitionIndices;
        std::vector<int> partitionModelOwner(clipMap->partitionCount, -1);
        std::vector<std::pair<unsigned, unsigned>> cmodelBrushCounts;
        std::vector<InlineModelCollisionSet> inlineModelCollisionSets;
        LeafBrushNodeCache leafBrushNodeCache;
        const auto worldAabbTreeRootIndices = GetWorldAabbTreeRootIndices(*clipMap);
        auto nonBrushPrimitiveCount = 0u;
        auto cmodelsWithBrushes = 0u;
        auto cmodelsWithOwnClipInfoBrushes = 0u;
        auto skippedOversizedInlineModels = 0u;
        auto duplicateNonBrushModelsSkipped = 0u;

        // CM_ClipHandleToModel @ .text:006A47E0 maps a model handle directly to
        // &cm.cmodels[handle] for handle < cm.numSubModels. cmodels[0] is the
        // world/special model, so inline brushmodels begin at index 1.
        for (auto cmodelIndex = 1u; clipMap->cmodels != nullptr && cmodelIndex < clipMap->numSubModels; cmodelIndex++)
        {
            const auto& cmodel = clipMap->cmodels[cmodelIndex];
            const auto modelIndex = cmodelIndex;
            const auto transform = inlineModelTransforms.find(modelIndex);

            // CMod_LoadSubmodelBrushNodes @ .text:0069CB80 receives firstBrush
            // and numBrushes for each inline model, then builds the cmodel's
            // leafBrushNode with CMod_PartionLeafBrushes @ .text:0069C9A0.
            // ResolveBrushModelBrushes keeps that recovery/fallback in one place.
            auto brushModelBrushes = ResolveBrushModelBrushes(leafBrushNodeCache, *clipMap, cmodel);
            if (brushModelBrushes.usesOwnClipInfo)
                cmodelsWithOwnClipInfoBrushes++;

            cmodelBrushCounts.emplace_back(modelIndex, static_cast<unsigned>(brushModelBrushes.indices.size()));
            if (brushModelBrushes.indices.size() > MAX_BRUSHES_PER_INLINE_MODEL)
            {
                skippedOversizedInlineModels++;
                continue;
            }

            // Non-brush ownership starts at cmodel.leaf. CM_Trace @ .text:006ADC30
            // passes cmodel + 0x20 to CM_TraceThroughLeaf @ .text:006A62C0 for
            // non-world model handles, proving cmodel_t embeds the cLeaf_s used
            // for both brush nodes and AABB collision.
            const auto rootAabbTreeIndices = GetAabbTreeIndicesForLeaf(*clipMap, cmodel.leaf);
            auto partitionIndices = CollectCollisionPartitionIndicesForRoots(*clipMap, rootAabbTreeIndices);

            // The engine's duplicate guard for non-brush triangles is keyed by
            // CollisionPartition. CM_TraceThroughAabbTree_work @ .text:006A1430
            // checks/stamps tw->partitionTraceStamp[partitionIndex], so exporter
            // ownership is partition-based instead of AABB-tree-based.
            ClaimCollisionPartitionsForModel(partitionIndices, partitionModelOwner, modelIndex);

            if (!rootAabbTreeIndices.empty() && partitionIndices.empty())
                duplicateNonBrushModelsSkipped++;

            const auto hasBrushGeometry =
                brushModelBrushes.clipInfo != nullptr && brushModelBrushes.clipInfo->brushes != nullptr && !brushModelBrushes.indices.empty();

            if (!hasBrushGeometry && partitionIndices.empty())
                continue;

            if (hasBrushGeometry)
                cmodelsWithBrushes++;

            inlineModelAabbTreeIndices.insert(inlineModelAabbTreeIndices.end(), rootAabbTreeIndices.begin(), rootAabbTreeIndices.end());
            inlineModelPartitionIndices.insert(inlineModelPartitionIndices.end(), partitionIndices.begin(), partitionIndices.end());
            if (UsesWorldBrushArray(*clipMap, brushModelBrushes.clipInfo))
            {
                // These brushes live in cm.info.brushes, so mark their world slots
                // as model-owned. They still remain in inlineModelCollisionSets so
                // the brushmodel itself is emitted with its map-ents transform.
                for (const auto brushIndex : brushModelBrushes.indices)
                {
                    if (brushIndex >= brushModelBrushes.clipInfo->numBrushes)
                        continue;

                    if (brushIndex < worldBrushModelOwner.size() && worldBrushModelOwner[brushIndex] < 0)
                        worldBrushModelOwner[brushIndex] = static_cast<int>(modelIndex);
                }
            }
            else if (brushModelBrushes.clipInfo != nullptr && brushModelBrushes.clipInfo->brushes != nullptr)
            {
                for (const auto brushIndex : brushModelBrushes.indices)
                {
                    if (brushIndex >= brushModelBrushes.clipInfo->numBrushes)
                        continue;

                    inlineModelBrushes.emplace_back(&brushModelBrushes.clipInfo->brushes[brushIndex]);
                }
            }

            inlineModelCollisionSets.push_back(
                {modelIndex,
                 brushModelBrushes.clipInfo,
                 transform != inlineModelTransforms.end() ? transform->second.origin : vec3_t{},
                 transform != inlineModelTransforms.end() ? transform->second.angles : vec3_t{},
                 std::move(brushModelBrushes.indices),
                 rootAabbTreeIndices,
                 std::move(partitionIndices),
                 std::move(brushModelBrushes.source)});
        }

        for (auto brushIndex = 0u; brushIndex < worldBrushModelOwner.size(); brushIndex++)
        {
            if (worldBrushModelOwner[brushIndex] < 0)
                continue;

            inlineModelWorldBrushIndices.emplace_back(brushIndex);
            inlineModelBrushes.emplace_back(&clipMap->info.brushes[brushIndex]);
        }

        std::sort(inlineModelWorldBrushIndices.begin(), inlineModelWorldBrushIndices.end());
        inlineModelWorldBrushIndices.erase(std::unique(inlineModelWorldBrushIndices.begin(), inlineModelWorldBrushIndices.end()), inlineModelWorldBrushIndices.end());
        std::sort(inlineModelAabbTreeIndices.begin(), inlineModelAabbTreeIndices.end());
        inlineModelAabbTreeIndices.erase(std::unique(inlineModelAabbTreeIndices.begin(), inlineModelAabbTreeIndices.end()), inlineModelAabbTreeIndices.end());
        std::sort(inlineModelPartitionIndices.begin(), inlineModelPartitionIndices.end());
        inlineModelPartitionIndices.erase(std::unique(inlineModelPartitionIndices.begin(), inlineModelPartitionIndices.end()), inlineModelPartitionIndices.end());

        // World non-brush also follows the AABB -> partition -> triIndices path
        // from CM_TraceThroughLeaf. Keep static world non-brush combined here;
        // partition/AABB identity is traversal data, not an object boundary.
        auto collisionTriangles = BuildNonBrushCollisionMeshes(
            *clipMap,
            "world",
            {},
            worldAabbTreeRootIndices,
            {},
            false,
            nullptr,
            &inlineModelPartitionIndices);
        nonBrushPrimitiveCount += static_cast<unsigned>(collisionTriangles.size());
        for (auto& primitive : collisionTriangles)
            primitives.emplace_back(std::move(primitive));

        AppendClipInfoBrushes(
            primitives,
            clipMap->info,
            "world",
            {},
            inlineModelBrushes,
            inlineModelWorldBrushIndices,
            nullptr,
            false,
            false,
            false);
        con::warn(
            "Clipmap \"{}\" has {} cmodels with brushes, {} cmodels with own clipinfo brushes, {} oversized inline models skipped, and {} inline model transforms from map ents",
            asset.m_name,
            cmodelsWithBrushes,
            cmodelsWithOwnClipInfoBrushes,
            skippedOversizedInlineModels,
            inlineModelTransforms.size());

        if (WRITE_DEBUG_FILE)
        {
            const auto debugName = asset.m_name + ".clipmap.debug.txt";
            const auto debugFile = context.OpenAssetFile(debugName);

            *debugFile << "asset=" << asset.m_name << "\n";
            *debugFile << "worldBrushes=" << clipMap->info.numBrushes << "\n";
            *debugFile << "numSubModels=" << clipMap->numSubModels << "\n";
            *debugFile << "cmodelsWithBrushes=" << cmodelsWithBrushes << "\n";
            *debugFile << "cmodelsWithOwnClipInfoBrushes=" << cmodelsWithOwnClipInfoBrushes << "\n";
            *debugFile << "skippedOversizedInlineModels=" << skippedOversizedInlineModels << "\n";
            *debugFile << "inlineModelTransforms=" << inlineModelTransforms.size() << "\n";
            *debugFile << "excludedWorldBrushes=" << inlineModelBrushes.size() << "\n";
            *debugFile << "excludedWorldBrushIndices=" << inlineModelWorldBrushIndices.size() << "\n";
            *debugFile << "modelOwnedWorldBrushSlots="
                       << std::count_if(worldBrushModelOwner.begin(), worldBrushModelOwner.end(), [](const int owner) { return owner >= 0; }) << "\n";
            *debugFile << "excludedWorldAabbTreeIndices=" << inlineModelAabbTreeIndices.size() << "\n";
            *debugFile << "inlineModelNonBrushAabbTreeIndices=" << inlineModelAabbTreeIndices.size() << "\n";
            *debugFile << "inlineModelNonBrushPartitions=" << inlineModelPartitionIndices.size() << "\n";
            *debugFile << "worldNonBrushAabbRoots=" << worldAabbTreeRootIndices.size() << "\n";
            *debugFile << "mergeCollisionTypes=" << (MERGE_COLLISION_TYPES ? "true" : "false") << "\n";
            *debugFile << "duplicateNonBrushModelsSkipped=" << duplicateNonBrushModelsSkipped << "\n";
            *debugFile << "worldLeafs=" << clipMap->numLeafs << "\n";
            *debugFile << "nonBrushPrimitiveCount=" << nonBrushPrimitiveCount << "\n";

            for (const auto& entry : inlineModelTransforms)
            {
                *debugFile << "transform model_*" << entry.first << " origin=" << entry.second.origin.x << " " << entry.second.origin.y << " " << entry.second.origin.z
                           << " angles=" << entry.second.angles.x << " " << entry.second.angles.y << " " << entry.second.angles.z << "\n";
            }

            for (const auto& entry : cmodelBrushCounts)
                *debugFile << "cmodel model_*" << entry.first << " brushes=" << entry.second << "\n";

            for (const auto& collisionSet : inlineModelCollisionSets)
            {
                *debugFile << "model_*" << collisionSet.modelIndex << " source=" << collisionSet.source << "\n";
                *debugFile << "model_*" << collisionSet.modelIndex << " recoveredBrushRange=";
                constexpr auto MAX_DEBUG_BRUSH_INDICES = 32u;
                const auto debugBrushIndexCount = std::min<unsigned>(MAX_DEBUG_BRUSH_INDICES, static_cast<unsigned>(collisionSet.brushIndices.size()));
                for (auto brushIndexOffset = 0u; brushIndexOffset < debugBrushIndexCount; brushIndexOffset++)
                {
                    if (brushIndexOffset > 0u)
                        *debugFile << ",";

                    *debugFile << collisionSet.brushIndices[brushIndexOffset];
                }
                if (collisionSet.brushIndices.size() > debugBrushIndexCount)
                    *debugFile << ",...";
                *debugFile << "\n";
            }
        }

        for (const auto& collisionSet : inlineModelCollisionSets)
        {
            InlineModelTransform transform{};
            transform.origin = collisionSet.origin;
            transform.angles = collisionSet.angles;
            // Model non-brush uses cmodel.leaf AABB roots as traversal, with
            // collisionSet.partitionIndices as the ownership filter. This mirrors
            // CM_Trace's model branch while avoiding duplicate world/model export.
            auto modelCollisionTriangles = BuildNonBrushCollisionMeshes(
                *clipMap,
                "model_*" + std::to_string(collisionSet.modelIndex),
                transform,
                collisionSet.aabbTreeIndices,
                {},
                false,
                &collisionSet.partitionIndices,
                nullptr);
            nonBrushPrimitiveCount += static_cast<unsigned>(modelCollisionTriangles.size());
            for (auto& primitive : modelCollisionTriangles)
                primitives.emplace_back(std::move(primitive));

            if (collisionSet.clipInfo != nullptr && collisionSet.clipInfo->brushes != nullptr && !collisionSet.brushIndices.empty())
            {
                AppendClipInfoBrushes(
                    primitives,
                    *collisionSet.clipInfo,
                    "model_*" + std::to_string(collisionSet.modelIndex),
                    {collisionSet.origin, collisionSet.angles},
                    {},
                    {},
                    &collisionSet.brushIndices,
                    true,
                    true);
            }
        }

        if (primitives.empty())
        {
            con::warn("Clipmap \"{}\" has no collision geometry to dump", asset.m_name);
            return;
        }

        const auto materialTextureLookup = BuildMaterialTextureLookup(context);
        WriteGlb(*glbFile, primitives, inlineModelTransforms, materialTextureLookup);
    }
} // namespace clip_map
