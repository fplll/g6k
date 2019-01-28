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
