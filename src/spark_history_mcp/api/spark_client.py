import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from urllib.parse import urljoin

import boto3
import requests
from pydantic import BaseModel

from spark_history_mcp.config.config import ServerConfig
from spark_history_mcp.models.spark_types import (
    ApplicationAttemptInfo,
    ApplicationEnvironmentInfo,
    ApplicationInfo,
    ExecutionData,
    ExecutorSummary,
    JobData,
    JobExecutionStatus,
    ProcessSummary,
    RDDStorageInfo,
    StageData,
    StageStatus,
    TaskData,
    TaskMetricDistributions,
    TaskStatus,
    ThreadStackTrace,
    VersionInfo,
)

T = TypeVar("T", bound=BaseModel)


class SparkRestClient:
    """
    Python client for the Spark REST API.
    """

    def __init__(self, server_config: ServerConfig):
        """
        Initialize the Spark REST client.

        Args:
            server_config: Configuration object
        """
        self.config = server_config
        self.base_url = self.config.url.rstrip("/") + "/api/v1"
        self.auth = None
        self.session = None
        self.use_proxy = self.config.use_proxy
        self.proxies = (
            self.use_proxy
            and {
                "http": "socks5h://localhost:8157",
                "https": "socks5h://localhost:8157",
            }
            or None
        )
        self.pattern = re.compile(r"(.*?/applications/[^/]+/)(.+)")

        # Determine whether to verify SSL certificates and timeout
        # Default to True for verify_ssl and 30 seconds for timeout if not specified
        self.verify_ssl = self.config.verify_ssl
        self.timeout = self.config.timeout

        # Set up authentication if provided
        if self.config.auth:
            if self.config.auth.username and self.config.auth.password:
                self.auth = (self.config.auth.username, self.config.auth.password)

    def _make_request(
        self, request_url: str, params: Optional[Dict[str, Any]]
    ) -> requests.Response:
        """
        Make a GET request to the Spark REST API.

        Args:
            request_url: The request URL
            params: Optional query parameters

        Returns:
            The response from the API
        """
        headers = {"Accept": "application/json"}

        # Add token to headers if provided
        if self.config.auth and self.config.auth.token:
            headers["Authorization"] = f"Bearer {self.config.auth.token}"

        # Use the verify_ssl setting for HTTPS requests
        verify = self.verify_ssl

        # Use the session if available, otherwise use requests directly
        if self.session:
            # Add headers to the session
            for key, value in headers.items():
                self.session.headers[key] = value

            response = self.session.get(
                request_url,
                params=params,
                timeout=self.timeout,
                verify=verify,
                proxies=self.proxies,
            )
        else:
            response = requests.get(
                request_url,
                params=params,
                headers=headers,
                auth=self.auth,
                timeout=self.timeout,
                verify=verify,
                proxies=self.proxies,
            )
        return response

    def _modify_url(self, url):
        match = self.pattern.search(url)
        if match:
            prefix = match.group(1)
            suffix = match.group(2)
            # Check if the suffix already starts with a number (attempt ID)
            if not re.match(r"^\d+/", suffix):
                # If no attempt ID present, add the first (and probably only) attempt of the app running on YARN
                app_attempt_id = 1
                return f"{prefix}{app_attempt_id}/{suffix}"
        return url

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make a GET request to the Spark REST API.

        Args:
            endpoint: The API endpoint to call
            params: Optional query parameters

        Returns:
            The JSON response from the API
        """
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))

        try:
            # Try original URL first
            first_response = self._make_request(url, params)
            first_response.raise_for_status()
            return first_response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404 and "/applications/" in url:
                modified_url = self._modify_url(url)
                try:
                    second_response = self._make_request(modified_url, params)
                    second_response.raise_for_status()
                    return second_response.json()
                except requests.exceptions.HTTPError as e2:
                    raise e2 from e  # Chain the exception with the original error
            # Raise the original error
            raise e from None

    def _parse_model(self, data: Dict[str, Any], model_class: Type[T]) -> T:
        """
        Parse JSON data into a Pydantic model.

        Args:
            data: The JSON data to parse
            model_class: The Pydantic model class to use

        Returns:
            An instance of the model class
        """
        return model_class.model_validate(data)

    def _parse_model_list(
        self, data: List[Dict[str, Any]], model_class: Type[T]
    ) -> List[T]:
        """
        Parse a list of JSON data into a list of Pydantic models.

        Args:
            data: The list of JSON data to parse
            model_class: The Pydantic model class to use

        Returns:
            A list of instances of the model class
        """
        return [self._parse_model(item, model_class) for item in data]

    def get_version(self) -> VersionInfo:
        """Get the Spark version."""
        data = self._get("version")
        return self._parse_model(data, VersionInfo)

    def list_applications(
        self,
        status: Optional[List[str]] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        min_end_date: Optional[str] = None,
        max_end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ApplicationInfo]:
        """
        Get a list of all applications.

        Args:
            status: Filter by application status (COMPLETED, RUNNING)
            min_date: Minimum start date (yyyy-MM-dd'T'HH:mm:ss.SSSz or yyyy-MM-dd)
            max_date: Maximum start date
            min_end_date: Minimum end date
            max_end_date: Maximum end date
            limit: Maximum number of applications to return

        Returns:
            List of ApplicationInfo objects
        """
        params = {}
        if status:
            params["status"] = status
        if min_date:
            params["minDate"] = min_date
        if max_date:
            params["maxDate"] = max_date
        if min_end_date:
            params["minEndDate"] = min_end_date
        if max_end_date:
            params["maxEndDate"] = max_end_date
        if limit:
            params["limit"] = limit

        data = self._get("applications", params)
        return self._parse_model_list(data, ApplicationInfo)

    def get_application(self, app_id: str) -> ApplicationInfo:
        """
        Get information about a specific application.

        Args:
            app_id: The application ID

        Returns:
            ApplicationInfo object
        """
        data = self._get(f"applications/{app_id}")
        return self._parse_model(data, ApplicationInfo)

    def get_application_attempt(
        self, app_id: str, attempt_id: str
    ) -> ApplicationAttemptInfo:
        """
        Get information about a specific application attempt.

        Args:
            app_id: The application ID
            attempt_id: The attempt ID

        Returns:
            ApplicationAttemptInfo object
        """
        data = self._get(f"applications/{app_id}/{attempt_id}")
        return self._parse_model(data, ApplicationAttemptInfo)

    def list_jobs(
        self, app_id: str, status: Optional[List[JobExecutionStatus]] = None
    ) -> List[JobData]:
        """
        Get a list of all jobs for an application.

        Args:
            app_id: The application ID
            status: Filter by job status

        Returns:
            List of JobData objects
        """
        params = {}
        if status:
            params["status"] = [s.value for s in status]

        data = self._get(f"applications/{app_id}/jobs", params)
        return self._parse_model_list(data, JobData)

    def get_job(self, app_id: str, job_id: int) -> JobData:
        """
        Get information about a specific job.

        Args:
            app_id: The application ID
            job_id: The job ID

        Returns:
            JobData object
        """
        data = self._get(f"applications/{app_id}/jobs/{job_id}")
        return self._parse_model(data, JobData)

    def list_stages(
        self,
        app_id: str,
        status: Optional[List[StageStatus]] = None,
        details: bool = False,
        with_summaries: bool = False,
        quantiles: str = "0.05, 0.25, 0.5, 0.75, 0.95",
        task_status: Optional[List[TaskStatus]] = None,
    ) -> List[StageData]:
        """
        Get a list of all stages for an application.

        Args:
            app_id: The application ID
            status: Filter by stage status
            details: Whether to include task details (WARNING: Setting this to True can significantly slow down the API call due to the large amount of task data returned)
            with_summaries: Whether to include summary metrics
            quantiles: Comma-separated list of quantiles to use for summary metrics
            task_status: Filter by task status (only takes effect when details=true)

        Returns:
            List of StageData objects
        """
        params = {
            "details": str(details).lower(),
            "withSummaries": str(with_summaries).lower(),
            "quantiles": quantiles,
        }

        if status:
            params["status"] = [s.value for s in status]
        # taskStatus parameter only takes effect when details=true
        if task_status and details:
            params["taskStatus"] = [s.value for s in task_status]

        data = self._get(f"applications/{app_id}/stages", params)
        return self._parse_model_list(data, StageData)

    def list_stage_attempts(
        self,
        app_id: str,
        stage_id: int,
        details: bool = False,  # Setting this to true is NOT recommended due to the amount of data returned.
        task_status: Optional[List[TaskStatus]] = None,
        with_summaries: bool = True,
        quantiles: str = "0.05, 0.25, 0.5, 0.75, 0.95",
    ) -> List[StageData]:
        """
        Get information about a specific stage.

        Args:
            app_id: The application ID
            stage_id: The stage ID
            details: Whether to include task details
            task_status: Filter by task status
            with_summaries: Whether to include summary metrics
            quantiles: Comma-separated list of quantiles to use for summary metrics

        Returns:
            List of StageData objects (one per attempt)
        """
        params = {
            "details": str(details).lower(),
            "withSummaries": str(with_summaries).lower(),
            "quantiles": quantiles,
        }

        if task_status:
            params["taskStatus"] = [s.value for s in task_status]

        data = self._get(f"applications/{app_id}/stages/{stage_id}", params)
        return self._parse_model_list(data, StageData)

    def get_stage_attempt(
        self,
        app_id: str,
        stage_id: int,
        attempt_id: int,
        details: bool = True,
        task_status: Optional[List[TaskStatus]] = None,
        with_summaries: bool = False,
        quantiles: str = "0.05, 0.25, 0.5, 0.75, 0.95",
    ) -> StageData:
        """
        Get information about a specific stage attempt.

        Args:
            app_id: The application ID
            stage_id: The stage ID
            attempt_id: The attempt ID
            details: Whether to include task details
            task_status: Filter by task status
            with_summaries: Whether to include summary metrics
            quantiles: Comma-separated list of quantiles to use for summary metrics

        Returns:
            StageData object
        """
        params = {
            "details": str(details).lower(),
            "withSummaries": str(with_summaries).lower(),
            "quantiles": quantiles,
        }

        if task_status:
            params["taskStatus"] = [s.value for s in task_status]

        data = self._get(
            f"applications/{app_id}/stages/{stage_id}/{attempt_id}", params
        )
        return self._parse_model(data, StageData)

    def get_stage_task_summary(
        self,
        app_id: str,
        stage_id: int,
        attempt_id: int,
        quantiles: str = "0.05, 0.25, 0.5, 0.75, 0.95",
    ) -> TaskMetricDistributions:
        """
        Get task summary metrics for a specific stage attempt.

        Args:
            app_id: The application ID
            stage_id: The stage ID
            attempt_id: The attempt ID
            quantiles: Comma-separated list of quantiles to use for summary metrics

        Returns:
            TaskMetricDistributions object
        """
        params = {"quantiles": quantiles}
        data = self._get(
            f"applications/{app_id}/stages/{stage_id}/{attempt_id}/taskSummary", params
        )
        return self._parse_model(data, TaskMetricDistributions)

    def list_stage_tasks(
        self,
        app_id: str,
        stage_id: int,
        attempt_id: int,
        offset: int = 0,
        length: int = 20,
        sort_by: str = "ID",
        status: Optional[List[TaskStatus]] = None,
    ) -> List[TaskData]:
        """
        Get tasks for a specific stage attempt.

        Args:
            app_id: The application ID
            stage_id: The stage ID
            attempt_id: The attempt ID
            offset: Pagination offset
            length: Number of tasks to return
            sort_by: Field to sort by
            status: Filter by task status

        Returns:
            List of TaskData objects
        """
        params = {"offset": offset, "length": length, "sortBy": sort_by}

        if status:
            params["status"] = [s.value for s in status]

        data = self._get(
            f"applications/{app_id}/stages/{stage_id}/{attempt_id}/taskList", params
        )
        return self._parse_model_list(data, TaskData)

    def list_executors(self, app_id: str) -> List[ExecutorSummary]:
        """
        Get a list of all executors for an application.

        Args:
            app_id: The application ID

        Returns:
            List of ExecutorSummary objects
        """
        data = self._get(f"applications/{app_id}/executors")
        return self._parse_model_list(data, ExecutorSummary)

    def list_all_executors(self, app_id: str) -> List[ExecutorSummary]:
        """
        Get a list of all executors (active and inactive) for an application.

        Args:
            app_id: The application ID

        Returns:
            List of ExecutorSummary objects
        """
        data = self._get(f"applications/{app_id}/allexecutors")
        return self._parse_model_list(data, ExecutorSummary)

    def list_executor_thread_dump(
        self, app_id: str, executor_id: str
    ) -> List[ThreadStackTrace]:
        """
        Get thread dump for a specific executor.

        Args:
            app_id: The application ID
            executor_id: The executor ID

        Returns:
            List of ThreadStackTrace objects
        """
        data = self._get(f"applications/{app_id}/executors/{executor_id}/threads")
        return self._parse_model_list(data, ThreadStackTrace)

    def get_task_thread_dump(
        self, app_id: str, task_id: int, executor_id: str
    ) -> ThreadStackTrace:
        """
        Get thread dump for a specific task.

        Args:
            app_id: The application ID
            task_id: The task ID
            executor_id: The executor ID

        Returns:
            ThreadStackTrace object
        """
        params = {"taskId": task_id, "executorId": executor_id}
        data = self._get(f"applications/{app_id}/threads", params)
        return self._parse_model(data, ThreadStackTrace)

    def list_all_processes(self, app_id: str) -> List[ProcessSummary]:
        """
        Get a list of all processes for an application.

        Args:
            app_id: The application ID

        Returns:
            List of ProcessSummary objects
        """
        data = self._get(f"applications/{app_id}/allmiscellaneousprocess")
        return self._parse_model_list(data, ProcessSummary)

    def list_rdds(self, app_id: str) -> List[RDDStorageInfo]:
        """
        Get a list of all RDDs for an application.

        Args:
            app_id: The application ID

        Returns:
            List of RDDStorageInfo objects
        """
        data = self._get(f"applications/{app_id}/storage/rdd")
        return self._parse_model_list(data, RDDStorageInfo)

    def get_rdd(self, app_id: str, rdd_id: int) -> RDDStorageInfo:
        """
        Get information about a specific RDD.

        Args:
            app_id: The application ID
            rdd_id: The RDD ID

        Returns:
            RDDStorageInfo object
        """
        data = self._get(f"applications/{app_id}/storage/rdd/{rdd_id}")
        return self._parse_model(data, RDDStorageInfo)

    def get_environment(self, app_id: str) -> ApplicationEnvironmentInfo:
        """
        Get environment information for an application.

        Args:
            app_id: The application ID

        Returns:
            ApplicationEnvironmentInfo object
        """
        data = self._get(f"applications/{app_id}/environment")
        return self._parse_model(data, ApplicationEnvironmentInfo)

    def get_metrics_prometheus(self, app_id: str) -> str:
        """
        Get Prometheus metrics for an application.

        Args:
            app_id: The application ID

        Returns:
            Prometheus metrics as a string
        """
        url = urljoin(
            self.base_url.replace("/api/v1", "/metrics/executors"), "prometheus"
        )

        if self.session:
            response = self.session.get(url, timeout=self.timeout, proxies=self.proxies)
        else:
            response = requests.get(url, timeout=self.timeout, proxies=self.proxies)

        response.raise_for_status()
        return response.text

    def get_sql_list(
        self,
        app_id: str,
        attempt_id: Optional[str] = None,
        details: bool = True,
        plan_description: bool = False,
        offset: int = 0,
        length: int = 20,
    ) -> List[ExecutionData]:
        """
        Get a list of all SQL executions for an application.

        Args:
            app_id: The application ID
            attempt_id: Optional attempt ID
            details: Whether to include execution details
            plan_description: Whether to include plan description
            offset: Pagination offset
            length: Number of executions to return

        Returns:
            List of ExecutionData objects
        """
        params = {
            "details": str(details).lower(),
            "planDescription": str(plan_description).lower(),
            "offset": offset,
            "length": length,
        }

        if attempt_id:
            endpoint = f"applications/{app_id}/{attempt_id}/sql"
        else:
            endpoint = f"applications/{app_id}/sql"

        data = self._get(endpoint, params)
        return [ExecutionData.from_dict(item) for item in data]

    def get_sql_execution(
        self,
        app_id: str,
        execution_id: int,
        attempt_id: Optional[str] = None,
        details: bool = True,
        plan_description: bool = True,
    ) -> ExecutionData:
        """
        Get information about a specific SQL execution.

        Args:
            app_id: The application ID
            execution_id: The execution ID
            attempt_id: Optional attempt ID
            details: Whether to include execution details
            plan_description: Whether to include plan description

        Returns:
            ExecutionData object
        """
        params = {
            "details": str(details).lower(),
            "planDescription": str(plan_description).lower(),
        }

        if attempt_id:
            endpoint = f"applications/{app_id}/{attempt_id}/sql/{execution_id}"
        else:
            endpoint = f"applications/{app_id}/sql/{execution_id}"

        data = self._get(endpoint, params)
        return ExecutionData.from_dict(data)

    def get_executor_log_content(
        self,
        app_id: str,
        executor_id: str,
        log_type: str = "stderr",
        offset: int = 0,
        length: int = 10000,
    ) -> str:
        """
        Retrieve actual log content from executor logs using Spark REST API.

        Args:
            app_id: The application ID
            executor_id: The executor ID
            log_type: Type of log ('stderr', 'stdout', 'log4j')
            offset: Starting byte offset
            length: Number of bytes to retrieve

        Returns:
            Raw log content as string

        Raises:
            ValueError: If executor or log type not found
            requests.exceptions.RequestException: If log retrieval fails
        """

        # Build Spark REST API endpoint for logs
        endpoint = f"/applications/{app_id}/executors/{executor_id}/logs/{log_type}"
        
        # Add pagination parameters if specified
        params = {}
        if offset > 0:
            params['offset'] = offset
        if length != 10000:
            params['length'] = length

        try:
            response = self._get(endpoint, params)
            return response if isinstance(response, str) else str(response)
            
        except Exception as e:
            error_msg = str(e)
            
            if "404" in error_msg or "Not Found" in error_msg:
                raise ValueError(
                    f"Executor '{executor_id}' or log type '{log_type}' not found. "
                    f"Common causes:\n"
                    f"• Executor doesn't exist in application '{app_id}'\n"
                    f"• Log aggregation was not enabled when the application ran\n"
                    f"• Logs have been cleaned up by retention policies\n"
                    f"• EMR/cluster configuration doesn't persist executor logs\n"
                    f"\nTip: Use 'get_application_logs_summary' to see what logs are actually available."
                ) from e
            else:
                raise ValueError(
                    f"Failed to retrieve log content for executor '{executor_id}' log type '{log_type}'. "
                    f"Error: {error_msg}. "
                    f"This could indicate network issues, authentication problems, or server errors."
                ) from e

    def _extract_emr_cluster_id(self) -> Optional[str]:
        """Extract EMR cluster ID from cluster ARN if available."""
        if not self.config.emr_cluster_arn:
            return None
        
        # EMR ARN format: arn:aws:elasticmapreduce:region:account:cluster/j-XXXXXXXXXXXXX
        arn_parts = self.config.emr_cluster_arn.split("/")
        if len(arn_parts) >= 2 and arn_parts[-1].startswith("j-"):
            return arn_parts[-1]  # Returns j-XXXXXXXXXXXXX
        return None

    def _extract_environment_from_cluster_name(self, cluster_name: str) -> Optional[str]:
        """
        Extract environment from cluster name using your organization's naming patterns.
        
        Supports patterns like:
        - cluster-name-prod
        - cluster-name-preprod  
        - cluster-name-dev
        - cluster-name-lab
        """
        if not cluster_name:
            return None
            
        # Environment suffixes (in priority order)
        env_patterns = [
            "prod",
            "preprod", 
            "dev",
            "lab"
        ]
        
        # Try to extract environment from end of cluster name
        cluster_lower = cluster_name.lower()
        
        for env in env_patterns:
            # Check for patterns: -env, _env, or just env at the end
            if cluster_lower.endswith(f"-{env}"):
                return env
            elif cluster_lower.endswith(f"_{env}"):
                return env
            elif cluster_lower.endswith(env) and len(cluster_lower) > len(env):
                # Make sure it's actually a suffix, not part of another word
                separator_pos = len(cluster_lower) - len(env) - 1
                if separator_pos >= 0 and cluster_lower[separator_pos] in ["-", "_"]:
                    return env
        
        return None

    def _discover_s3_log_configuration(self) -> Dict[str, Optional[str]]:
        """
        Auto-discover S3 log configuration from EMR cluster if possible.
        
        Returns:
            Dict with 'bucket', 'path_pattern', 'environment', and 'cluster_name' keys
        """
        result = {
            "bucket": None, 
            "path_pattern": None, 
            "environment": None,
            "cluster_name": None
        }
        
        cluster_id = self._extract_emr_cluster_id()
        if not cluster_id:
            return result
            
        try:
            # Try to get EMR cluster configuration with profile if configured
            region = self.config.emr_cluster_arn.split(":")[3]
            if self.config.aws_profile:
                session = boto3.Session(profile_name=self.config.aws_profile)
                emr_client = session.client("emr", region_name=region)
            else:
                emr_client = boto3.client("emr", region_name=region)
            
            # Get cluster details
            response = emr_client.describe_cluster(ClusterId=cluster_id)
            cluster = response.get("Cluster", {})
            
            # Extract cluster name and environment
            cluster_name = cluster.get("Name", "")
            result["cluster_name"] = cluster_name
            
            environment = self._extract_environment_from_cluster_name(cluster_name)
            result["environment"] = environment
            
            # Look for log URI in cluster configuration
            log_uri = cluster.get("LogUri")
            if log_uri and log_uri.startswith("s3://"):
                # Extract bucket from log URI like s3://bucket-name/path/
                log_parts = log_uri.replace("s3://", "").split("/", 1)
                result["bucket"] = log_parts[0]
                
                # Try to detect existing path pattern first
                detected_pattern = None
                if len(log_parts) > 1:
                    base_path = log_parts[1].rstrip("/") + "/"
                    # Check if path contains cluster ID to build pattern
                    if cluster_id in base_path:
                        detected_pattern = base_path.replace(cluster_id, "{cluster_id}")
                        # If environment was detected and path doesn't include it, enhance the pattern
                        if environment and environment not in detected_pattern:
                            # Try to insert environment into common patterns
                            if "elasticmapreduce/{cluster_id}" in detected_pattern:
                                detected_pattern = detected_pattern.replace(
                                    "elasticmapreduce/{cluster_id}",
                                    f"elasticmapreduce/{environment}/{{cluster_id}}"
                                )
                
                # Use detected pattern or build environment-aware pattern
                if detected_pattern:
                    result["path_pattern"] = detected_pattern
                elif environment:
                    # Environment-aware patterns based on common conventions
                    env_patterns = [
                        f"emr_clusters/{environment}/{{cluster_id}}/containers/",  # Most common
                        f"elasticmapreduce/{environment}/{{cluster_id}}/containers/",  # AWS with env
                        f"{environment}/elasticmapreduce/{{cluster_id}}/containers/",  # Env first
                        f"emr/{environment}/{{cluster_id}}/containers/",  # Short form
                    ]
                    result["path_pattern"] = env_patterns[0]  # Use most common as default
                else:
                    # Fallback to standard patterns
                    result["path_pattern"] = "elasticmapreduce/{cluster_id}/containers/"
                
        except Exception:
            # Auto-discovery failed, will fall back to defaults/config
            pass
            
        return result

    def _get_s3_log_configuration(self) -> Dict[str, Optional[str]]:
        """
        Get S3 log configuration using layered approach:
        1. Explicit configuration (highest priority)
        2. Auto-discovery from EMR
        3. Smart defaults
        """
        config = {"bucket": None, "path_pattern": None}
        
        # Layer 1: Explicit configuration (highest priority)
        if self.config.s3_log_bucket:
            config["bucket"] = self.config.s3_log_bucket
            config["path_pattern"] = (
                self.config.s3_log_path_pattern or 
                "elasticmapreduce/{cluster_id}/containers/"
            )
            return config
        
        # Layer 2: Auto-discovery
        discovered = self._discover_s3_log_configuration()
        if discovered["bucket"]:
            config["bucket"] = discovered["bucket"]
            config["path_pattern"] = (
                discovered["path_pattern"] or 
                "elasticmapreduce/{cluster_id}/containers/"
            )
            return config
            
        # Layer 3: Smart defaults (AWS standard patterns)
        cluster_id = self._extract_emr_cluster_id()
        if cluster_id:
            # Try common AWS account patterns
            region = self.config.emr_cluster_arn.split(":")[3]
            account_id = self.config.emr_cluster_arn.split(":")[4]
            
            config["bucket"] = f"aws-logs-{account_id}-{region}"
            config["path_pattern"] = "elasticmapreduce/{cluster_id}/containers/"
            
        return config

    def _build_s3_log_path(self, app_id: str, executor_id: str, log_type: str = "stderr") -> Optional[str]:
        """Build S3 path for executor logs using smart configuration discovery."""
        cluster_id = self._extract_emr_cluster_id()
        if not cluster_id:
            return None
        
        config = self._get_s3_log_configuration()
        if not config["bucket"]:
            return None
        
        # Build the full path with correct EMR container naming
        path = config["path_pattern"].format(cluster_id=cluster_id)
        
        # EMR containers follow pattern: container_{app_id}_01_{executor_padded}
        # Extract numeric part from app_id and executor_id for container naming
        app_numeric = app_id.replace("application_", "")
        
        # Safely parse executor_id - handle both numeric and string formats
        try:
            if isinstance(executor_id, str) and executor_id.isdigit():
                executor_num = int(executor_id)
            elif isinstance(executor_id, int):
                executor_num = executor_id
            else:
                # Handle executor IDs like "driver", "1", etc.
                # For driver, use 1; for others try to extract numbers
                if executor_id.lower() == "driver":
                    executor_num = 1
                else:
                    # Extract any digits from the executor_id
                    import re
                    digits = re.findall(r'\d+', str(executor_id))
                    executor_num = int(digits[0]) if digits else 1
            
            executor_padded = f"{executor_num:06d}"  # Convert to 6-digit padded format
        except (ValueError, IndexError) as e:
            raise ValueError(f"Cannot parse executor_id '{executor_id}': {e}. Expected numeric value or 'driver'.")
        container_name = f"container_{app_numeric}_01_{executor_padded}"
        
        # Logs are gzipped in EMR
        log_file = f"{log_type}.gz"
        
        s3_path = f"s3://{config['bucket']}/{path}{app_id}/{container_name}/{log_file}"
        return s3_path

    def get_executor_log_content_from_s3(
        self,
        app_id: str,
        executor_id: str,
        log_type: str = "stderr",
        max_lines: int = 1000,
    ) -> str:
        """
        Retrieve executor log content from S3 for EMR clusters.
        
        This method accesses logs directly from S3 when they're not available
        through the Spark History Server API, which is common for terminated EMR clusters.

        Args:
            app_id: The application ID
            executor_id: The executor ID
            log_type: Type of log ('stderr', 'stdout')
            max_lines: Maximum number of lines to retrieve from end of file

        Returns:
            Raw log content as string

        Raises:
            ValueError: If S3 access is not configured or logs not found
        """
        s3_path = self._build_s3_log_path(app_id, executor_id, log_type)
        if not s3_path:
            raise ValueError(
                "S3 log access not configured. Required: emr_cluster_arn and s3_log_bucket in server config."
            )
        
        try:
            # Initialize S3 client with profile if configured
            region = self.config.emr_cluster_arn.split(":")[3] if self.config.emr_cluster_arn else "us-east-1"
            if self.config.aws_profile:
                session = boto3.Session(profile_name=self.config.aws_profile)
                s3_client = session.client("s3", region_name=region)
            else:
                s3_client = boto3.client("s3", region_name=region)
            
            # Parse S3 path
            s3_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket = s3_parts[0]
            key = s3_parts[1]
            
            # Try to get object (handle gzipped content)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content_bytes = response['Body'].read()
            
            # Handle gzipped content
            if key.endswith('.gz'):
                import gzip
                content = gzip.decompress(content_bytes).decode('utf-8', errors='replace')
            else:
                content = content_bytes.decode('utf-8', errors='replace')
            
            # Return last N lines if content is long
            lines = content.splitlines()
            if len(lines) > max_lines:
                content = '\n'.join(lines[-max_lines:])
                content = f"... (showing last {max_lines} lines of {len(lines)} total lines)\n\n" + content
            
            return content
            
        except Exception as e:
            error_msg = str(e)
            if "NoSuchKey" in error_msg or "Not Found" in error_msg:
                raise ValueError(
                    f"Log file not found in S3: {s3_path}\n"
                    f"This could mean:\n"
                    f"• Logs were not configured to be stored in S3\n"
                    f"• Log retention policy has cleaned up the files\n"
                    f"• The S3 path pattern is incorrect for your EMR setup"
                ) from e
            elif "NoSuchBucket" in error_msg:
                raise ValueError(
                    f"S3 bucket not accessible: {bucket}\n"
                    f"Check that the bucket exists and you have read permissions."
                ) from e
            else:
                raise ValueError(
                    f"Failed to access S3 logs: {error_msg}\n"
                    f"S3 path: {s3_path}"
                ) from e

    def get_executor_log_content_hybrid(
        self,
        app_id: str,
        executor_id: str,
        log_type: str = "stderr",
        offset: int = 0,
        length: int = 10000,
    ) -> Dict[str, Any]:
        """
        Hybrid method that tries History Server API first, then falls back to S3.
        
        Returns both content and metadata about the source.
        """
        result = {
            "application_id": app_id,
            "executor_id": executor_id,
            "log_type": log_type,
            "content": "",  # Initialize as empty string, not None
            "source": None,
            "error": None
        }
        
        # Try History Server API first
        try:
            content = self.get_executor_log_content(app_id, executor_id, log_type, offset, length)
            result["content"] = content or ""  # Ensure content is never None
            result["source"] = "spark_history_server_api"
            return result
        except Exception as api_error:
            result["error"] = f"API access failed: {str(api_error)}"
            
        # Fallback to S3 if configured
        if self.config.emr_cluster_arn and self.config.s3_log_bucket:
            try:
                # Convert length to approximate lines for S3 access
                estimated_lines = max(length // 120, 100)
                content = self.get_executor_log_content_from_s3(
                    app_id, executor_id, log_type, max_lines=estimated_lines
                )
                result["content"] = content or ""  # Ensure content is never None
                result["source"] = "s3_direct_access"
                result["error"] = None  # Clear the API error since S3 worked
                return result
            except Exception as s3_error:
                result["error"] += f" | S3 fallback failed: {str(s3_error)}"
        else:
            result["error"] += " | S3 fallback not configured (missing emr_cluster_arn or s3_log_bucket)"
        
        return result

    def get_application_logs_summary(self, app_id: str) -> Dict[str, Any]:
        """
        Get summary of all available logs for an application.

        Args:
            app_id: The application ID

        Returns:
            Dictionary with executor IDs and their available log types
        """
        executors = self.list_all_executors(app_id=app_id)

        logs_summary = {
            "application_id": app_id,
            "total_executors": len(executors),
            "executors_with_logs": 0,
            "log_types_available": set(),
            "executor_logs": {}
        }

        for executor in executors:
            if executor.executor_logs:
                logs_summary["executors_with_logs"] += 1
                logs_summary["log_types_available"].update(executor.executor_logs.keys())
                logs_summary["executor_logs"][executor.id] = {
                    "log_types": list(executor.executor_logs.keys()),
                    "log_urls": executor.executor_logs,
                    "host_port": executor.host_port,
                    "is_active": executor.is_active
                }

        logs_summary["log_types_available"] = sorted(list(logs_summary["log_types_available"]))

        return logs_summary

    def search_executor_logs(
        self,
        app_id: str,
        search_pattern: str,
        log_type: str = "stderr",
        max_executors: int = 5,
        max_lines_per_executor: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search for patterns across executor logs.

        Args:
            app_id: The application ID
            search_pattern: Text pattern to search for (case-insensitive)
            log_type: Type of log to search ('stderr', 'stdout', 'log4j')
            max_executors: Maximum number of executors to search
            max_lines_per_executor: Maximum lines to search per executor

        Returns:
            List of matches with executor context
        """
        logs_summary = self.get_application_logs_summary(app_id)
        
        if logs_summary["executors_with_logs"] == 0:
            return []

        matches = []
        search_pattern_lower = search_pattern.lower()
        
        # Get executors that have the requested log type
        executor_ids = [
            exec_id for exec_id, exec_info in logs_summary["executor_logs"].items()
            if log_type in exec_info["log_types"]
        ][:max_executors]

        for executor_id in executor_ids:
            try:
                # Estimate bytes needed for max_lines_per_executor
                estimated_bytes = max_lines_per_executor * 150  # ~150 chars per line
                
                log_content = self.get_executor_log_content(
                    app_id=app_id,
                    executor_id=executor_id,
                    log_type=log_type,
                    length=estimated_bytes
                )

                lines = log_content.splitlines()
                for line_num, line in enumerate(lines, 1):
                    if search_pattern_lower in line.lower():
                        matches.append({
                            "executor_id": executor_id,
                            "line_number": line_num,
                            "line_content": line.strip(),
                            "log_type": log_type
                        })

                        # Limit total matches to prevent overwhelming results
                        if len(matches) >= 50:
                            break

            except Exception as e:
                # Continue with other executors if one fails
                matches.append({
                    "executor_id": executor_id,
                    "error": f"Failed to search logs: {str(e)}",
                    "log_type": log_type
                })
                continue

            # Break if we have enough matches
            if len(matches) >= 50:
                break

        return matches
