//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef CTOR_HOOK_H
#define CTOR_HOOK_H 1

class Graph;

namespace hnnx {
class OpIoPtrs;

template <typename T> inline void ctor_hook(Graph &, T &ref)
{
    return;
}

// the 'pre-hook' can install an OpHookBase pointer into the op_io_ptrs
// default is to do nothing.
template <typename T> inline void ctor_ophook(OpIoPtrs const &op_io_ptrs)
{
    return;
}

} // namespace hnnx

#ifdef PREPARE_DISABLED
#define CTOR_HOOK(FUNC, VAR, CODE)
#else
#define CTOR_HOOK(FUNC, VAR, CODE)                                                                                     \
    template <>                                                                                                        \
    [[maybe_unused]] inline void hnnx::ctor_hook(Graph &graph_in, typename DerivedType<(&FUNC)>::type &VAR)            \
    {                                                                                                                  \
        CODE                                                                                                           \
    }
#endif

// maybe we could add more than one ophook... just define this with different #'s of parms.
// 'HOOKCLASS' must be a subclass of OpHookBase, which defines the hook.
#define CTOR_OPHOOK(FUNC, HOOKCLASS)                                                                                   \
    template <> inline void hnnx::ctor_ophook<typename DerivedType<(&FUNC)>::type>(OpIoPtrs const &op_io_ptrs)         \
    {                                                                                                                  \
        static constexpr HOOKCLASS hook;                                                                               \
        const_cast<OpIoPtrs &>(op_io_ptrs).add_ophook(&hook);                                                          \
    }

#endif
