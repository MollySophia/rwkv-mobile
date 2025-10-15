//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef GRAPH_HANDLE_DEFS_H
#define GRAPH_HANDLE_DEFS_H 1

#include <type_traits>
#include "forward_classes.h"

namespace gHN {
class gH;
class gX;
} // namespace gHN

namespace hnnx {
class GraphHandleBase;
}
namespace graph_handle_private {
Graph const &graph_from(hnnx::GraphHandleBase const &);
}

//
// This header does not require 'graph.h'
//
namespace hnnx {

// Any subclass of GraphHandleBase may be used as 'pass-by-value' parameter in an Op function.
// The class hierarchy is defined in this header.
//
class GraphHandleBase {
    friend Graph const &graph_handle_private::graph_from(hnnx::GraphHandleBase const &);

  protected:
    Graph const *m_graphp;
    explicit GraphHandleBase(Graph const &g) : m_graphp(&g) {}

    inline Graph const &graph_ref() const { return *m_graphp; }

    // All these are protected, since only subclasses may actually exist.
    ~GraphHandleBase() = default;
    GraphHandleBase(GraphHandleBase const &) = default;
    GraphHandleBase(GraphHandleBase &&) = default;
    GraphHandleBase &operator=(GraphHandleBase const &) = delete;
    GraphHandleBase &operator=(GraphHandleBase &&) = delete;
};

using GHandle = gHN::gH; // hnnx::GHandle
using GHandleX = gHN::gX; // hnnx::GHandleX
} // namespace hnnx

////////////////////
// Note: GHandle and GHandleX are the ones which appear in all the 'Op' functions,
// and are therefore mangled into everything. They are thus given short names
// gHN::gH and gHN::gX, and 'typedef'd to hnnx::GHandle and hnnx::GHandleX.

class gHN::gH : public hnnx::GraphHandleBase {
  public:
    gH(Graph const &g) : hnnx::GraphHandleBase(g) {}
    ~gH() = default;
    gH(gH const &) = default;
    gH(gH &&) = default;
    gH &operator=(gH const &) = delete;
    gH &operator=(gH &&) = delete;

    // This being false, means that adding a parameter of this type to the op function has
    // no effect on the "op id" string used in the pickle.
    static constexpr bool appears_in_opid_string = false;
};

// GHandleX is effectively identical to GHandle, but a '.' will appear at the end the Op id string
// if this is used as an Op function parameter.
// The two types will copy-construct from each other.
class gHN::gX : public hnnx::GHandle {
  public:
    gX(Graph const &g) : hnnx::GHandle(g) {}
    inline ~gX() = default;
    gX(gX const &) = default;
    gX(gX &&) = default;
    gX &operator=(gX const &) = delete;
    gX &operator=(gX &&) = delete;

    // It is ok to copy-construct this from GHandle, too.
    inline gX(hnnx::GHandle const &gh) : hnnx::GHandle(gh) {}

    // This being true means the parameter type appears in opid string used in pickle;
    // but for now, only as "" (meaning there will be a trailing ".").
    static constexpr bool appears_in_opid_string = true;
};
///////////////////////

#if 1 // will be excluded in some situations.
inline Graph const &graph_handle_private::graph_from(hnnx::GraphHandleBase const &gh)
{
    return gh.graph_ref();
}
#endif

// GHANDLE_ACCESS_FUNC_QUALIFIER will change to API_EXPORT in some situations
#define GHANDLE_ACCESS_FUNC_QUALIFIER
// changes to 'nothing' in some situations.
#define GHANDLE_ACCESS_FUNC_INLINE inline

#if !defined(GHANDLE_INCLUDE_IMPL)
#define GHANDLE_INCLUDE_IMPL 1
#endif

#endif // GRAPH_HANDLE_DEFS_H
