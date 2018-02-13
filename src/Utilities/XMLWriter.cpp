////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Mark Dewing, mdewing@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Mark Dewing, mdewing@anl.gov,
//    Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////

/** @file XMLWriter.cpp
 * @brief Helper functions for tiny xml2
 */
#include "Utilities/XMLWriter.h"

namespace qmcplusplus
{

XMLNode *
MakeTextElement(XMLDocument &doc, const std::string &name, const std::string &value)
{
  XMLNode* name_node = doc.NewElement(name.c_str());
  XMLText* value_node = doc.NewText("");
  value_node->SetValue(value.c_str());
  name_node->InsertEndChild(value_node);
  return name_node;
}
}
