/***\
*
*   Copyright (C) 2018-2021 Team G6K
*
*   This file is part of G6K. G6K is free software:
*   you can redistribute it and/or modify it under the terms of the
*   GNU General Public License as published by the Free Software Foundation,
*   either version 2 of the License, or (at your option) any later version.
*
*   G6K is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with G6K. If not, see <http://www.gnu.org/licenses/>.
*
****/


#include "siever.h"

bool Siever::set_params(const SieverParams &params)
{
  this->params = params;
  reserve(params.reserved_db_size);
  set_threads(params.threads);

  if(this->params.bgj1_transaction_bulk_size == 0)
  {
    this->params.bgj1_transaction_bulk_size = 10 + 2*this->params.threads;
  }
  return true;
}

SieverParams Siever::get_params()
{
  return this->params;
}
