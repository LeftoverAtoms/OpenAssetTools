#include "GfxWorldDumperT6.h"

#include <format>

using namespace T6;

namespace
{
    constexpr auto GFX_WORLD_VERTEX_STRIDE = 36;

    const vec3_t& GetSurfaceVertexPosition(const GfxWorld& world, const GfxSurface& surface, const unsigned vertIndex)
    {
        const auto* vertexData = reinterpret_cast<const unsigned char*>(world.draw.vd1.data);
        return *reinterpret_cast<const vec3_t*>(vertexData + surface.tris.vertexDataOffset0 + GFX_WORLD_VERTEX_STRIDE * vertIndex);
    }

    void WriteSurfaceVertices(std::ostream& stream, const GfxWorld& world, const GfxSurface& surface)
    {
        for (auto vertIndex = 0; vertIndex < surface.tris.vertexCount; vertIndex++)
        {
            const auto& pos = GetSurfaceVertexPosition(world, surface, vertIndex);
            stream << pos.x << " " << pos.y << " " << pos.z << "\n";
        }
    }

    bool WriteWorldVertices(std::ostream& stream, const GfxWorld& world)
    {
        if (world.dpvs.surfaces == nullptr || world.draw.vd1.data == nullptr)
            return false;

        for (auto surfaceIndex = 0; surfaceIndex < world.surfaceCount; surfaceIndex++)
        {
            const auto& surface = world.dpvs.surfaces[surfaceIndex];
            WriteSurfaceVertices(stream, world, surface);
        }

        return true;
    }
} // namespace

namespace gfx_world
{
    void DumperT6::DumpAsset(AssetDumpingContext& context, const XAssetInfo<AssetGfxWorld::Type>& asset)
    {
        const auto* world = asset.Asset();
        const auto assetFile = context.OpenAssetFile(std::format("model_export/{}.vertices", asset.m_name));
        if (!assetFile)
            return;

        WriteWorldVertices(*assetFile, *world);
    }
} // namespace gfx_world
