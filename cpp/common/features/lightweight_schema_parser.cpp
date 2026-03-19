//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <common/features/lightweight_schema_parser.h>

#include <fstream>
#include <filesystem>
#include <cerrno>
#include <cstring>

#include <boost/algorithm/string.hpp>
#include <boost/spirit/home/x3.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <common/features/lightweight_feature_compute_exec.h>

namespace metaspore {

using boost::spirit::x3::_attr;
using boost::spirit::x3::char_;
namespace fs = std::filesystem;

// identifier rule for table name or column name
static auto ident = +(char_("a-zA-z_0-9"));

status LightweightSchemaParser::parse(const std::string &file, LightweightFeatureComputeExec &exec) {
    std::ifstream ifs(file);
    if (!ifs) {
        return absl::InternalError(fmt::format("Open file {} failed {}", file, strerror(errno)));
    }
    int feature_count = 0;
    auto status = parse_hash_and_combine(ifs, exec, feature_count);
    if (!status.ok()) {
        return status;
    }
    if (exec.get_input_names().empty()) {
        CALL_AND_RETURN_IF_STATUS_NOT_OK(parse_table_name_from_path(file, exec));
    }
    return absl::OkStatus();
}

status LightweightSchemaParser::parse_table_name_from_path(const std::string &file,
                                                           LightweightFeatureComputeExec &exec) {
    auto dir_name = fs::path(file).parent_path().filename().string();
    auto name = dir_name;
    boost::trim_if(name, boost::is_any_of("/\\"));
    if (name.empty()) {
        return absl::InvalidArgumentError(fmt::format("{} cannot be a valid table name", dir_name));
    }
    if (boost::starts_with(name, "sparse_")) {
        auto subname = name.substr(7);
        if (subname.empty()) {
            return absl::InvalidArgumentError(fmt::format(
                "{} should contain a table name after sparse_ to be a valid table name", dir_name));
        }
        name = subname;
    }
    return exec.add_source(name);
}

status LightweightSchemaParser::parse_hash_and_combine(std::istream &is,
                                                       LightweightFeatureComputeExec &exec,
                                                       int &feature_count) {
    std::string line;
    std::vector<std::string> feature_columns;
    auto push = [&](auto &context) { feature_columns.push_back(_attr(context)); };
    auto combine_rule = ident[push] % '#';
    std::vector<std::vector<std::string>> projections;

    while (std::getline(is, line)) {
        feature_columns.clear();
        boost::trim_if(line, boost::is_any_of("# \t\n\r"));
        if (line.empty())
            break;

        if (boost::spirit::x3::parse(line.begin(), line.end(), combine_rule)) {
            if (feature_columns.empty())
                continue;
            projections.push_back(feature_columns);
            feature_count++;
        } else {
            auto m = fmt::format("Parsing combine rule failed {}", line);
            spdlog::error(m);
            return absl::InvalidArgumentError(m);
        }
    }
    return exec.add_projection(std::move(projections));
}

} // namespace metaspore
