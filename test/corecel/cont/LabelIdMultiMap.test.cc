//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/LabelIdMultiMap.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/LabelIdMultiMap.hh"

#include <iostream>
#include <sstream>

#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//

using CatId = OpaqueId<struct Cat>;
using CatMultiMap = LabelIdMultiMap<CatId>;
using VecLabel = CatMultiMap::VecLabel;

std::ostream& operator<<(std::ostream& os, CatId const& cat)
{
    os << "CatId{";
    if (cat)
        os << cat.unchecked_get();
    os << "}";
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace

TEST(LabelTest, ordering)
{
    EXPECT_EQ(Label("a"), Label("a"));
    EXPECT_EQ(Label("a", "1"), Label("a", "1"));
    EXPECT_NE(Label("a"), Label("b"));
    EXPECT_NE(Label("a", "1"), Label("a", "2"));
    EXPECT_TRUE(Label("a") < Label("b"));
    EXPECT_FALSE(Label("a") < Label("a"));
    EXPECT_FALSE(Label("b") < Label("a"));
    EXPECT_TRUE(Label("a") < Label("a", "1"));
    EXPECT_TRUE(Label("a", "0") < Label("a", "1"));
    EXPECT_FALSE(Label("a", "1") < Label("a", "1"));
    EXPECT_FALSE(Label("a", "2") < Label("a", "1"));
}

TEST(LabelTest, construction)
{
    EXPECT_EQ(Label("foo"), Label::from_geant("foo"));
    EXPECT_EQ(Label("foo", "0xdeadb01d"), Label::from_geant("foo0xdeadb01d"));
    EXPECT_EQ(Label("foo", "0x1234"), Label::from_geant("foo0x1234"));
    EXPECT_EQ(Label("foo", "0x1e0cea00x1e0c5c0"),
              Label::from_geant("foo0x1e0cea00x1e0c5c0"));
    EXPECT_EQ(Label("foo", "0x1e0c8c0_refl"),
              Label::from_geant("foo0x1e0c8c0_refl"));

    EXPECT_EQ(Label("bar"), Label::from_separator("bar", '@'));
    EXPECT_EQ(Label("bar"), Label::from_separator("bar@", '@'));
    EXPECT_EQ(Label("bar", "123"), Label::from_separator("bar@123", '@'));
}

TEST(LabelTest, output)
{
    std::ostringstream os;
    os << Label{"bar", "123"};
    EXPECT_EQ("bar@123", os.str());
}

//---------------------------------------------------------------------------//
TEST(LabelIdMultiMapTest, empty)
{
    const CatMultiMap default_cats;
    const CatMultiMap empty_cats(VecLabel{});
    for (CatMultiMap const* cats : {&default_cats, &empty_cats})
    {
        EXPECT_EQ(0, cats->size());
        EXPECT_EQ(CatId{}, cats->find(Label{"merry"}));
        EXPECT_EQ(0, cats->find_all("pippin").size());
#if CELERITAS_DEBUG
        EXPECT_THROW(cats->get(CatId{0}), DebugError);
#endif
    }
}

TEST(LabelIdMultiMapTest, no_ext_with_duplicates)
{
    CatMultiMap cats{VecLabel{{"dexter", "andy", "loki", "", "", ""}}};
    EXPECT_EQ(6, cats.size());
    EXPECT_EQ(CatId{}, cats.find("nyoka"));
    EXPECT_EQ(CatId{0}, cats.find("dexter"));
    EXPECT_EQ(CatId{1}, cats.find("andy"));
    EXPECT_EQ(CatId{2}, cats.find("loki"));
    EXPECT_EQ(CatId{2}, cats.find(Label{"loki"}));

    static const CatId expected_duplicates[] = {CatId{3}, CatId{4}, CatId{5}};
    EXPECT_VEC_EQ(expected_duplicates, cats.duplicates());
}

TEST(LabelIdMultiMapTest, some_labels)
{
    CatMultiMap cats{{{Label{"leroy"},
                       Label{"fluffy"},
                       {"fluffy", "jr"},
                       {"fluffy", "sr"}}}};
    EXPECT_EQ(4, cats.size());
    EXPECT_EQ(CatId{1}, cats.find(Label{"fluffy"}));
    EXPECT_EQ(CatId{2}, cats.find(Label{"fluffy", "jr"}));
    {
        auto found = cats.find_all("fluffy");
        static const CatId expected_found[] = {CatId{1}, CatId{2}, CatId{3}};
        EXPECT_VEC_EQ(expected_found, found);
    }
}

TEST(LabelIdMultiMapTest, shuffled_labels)
{
    const std::vector<Label> labels = {
        {"c", "2"},
        {"b", "1"},
        {"b", "0"},
        {"a", "0"},
        {"c", "0"},
        {"c", "1"},
    };

    CatMultiMap cats{labels};

    // Check ordering of IDs
    for (auto i : range<CatId::size_type>(labels.size()))
    {
        EXPECT_EQ(labels[i], cats.get(CatId{i}));
    }

    // Check discontinuous ID listing
    {
        auto found = cats.find_all("a");
        static const CatId expected_found[] = {CatId{3}};
        EXPECT_VEC_EQ(expected_found, found);
    }
    {
        auto found = cats.find_all("b");
        static const CatId expected_found[] = {CatId{2}, CatId{1}};
        EXPECT_VEC_EQ(expected_found, found);
    }
    {
        auto found = cats.find_all("c");
        static const CatId expected_found[] = {CatId{4}, CatId{5}, CatId{0}};
        EXPECT_VEC_EQ(expected_found, found);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
