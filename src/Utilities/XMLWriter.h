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

/** @file XMLWriter.h
 */

#ifndef QMCPLUSPLUS_XML_WRITER_H
#define QMCPLUSPLUS_XML_WRITER_H

#include <tinyxml/tinyxml2.h>
#include <string>

namespace qmcplusplus
{

using tinyxml2::XMLNode;
using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLText;


XMLNode *MakeTextElement(XMLDocument &doc, const std::string &name, const std::string &value);

}

#endif
